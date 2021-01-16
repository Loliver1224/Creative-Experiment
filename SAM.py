"""
Copyright (c) 2020 Yutaro Ogawa
"""
from numpy.random import *
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
from sklearn.preprocessing import scale
from cdt.utils.torch import ChannelBatchNorm1d, MatrixSampler, Linear3D
import torch
import torch.nn as nn
import random
import numpy as np

np.random.seed(1234)
random.seed(1234)
torch.autograd.set_detect_anomaly(True)


class SAMDiscriminator(nn.Module):
    """SAMのDiscriminatorのニューラルネットワーク
    """

    def __init__(self, nfeatures, dnh, hlayers):
        super(SAMDiscriminator, self).__init__()

        # ----------------------------------
        # ネットワークの用意
        # ----------------------------------
        self.nfeatures = nfeatures  # 入力変数の数

        layers = []
        layers.append(nn.Linear(nfeatures, dnh))
        layers.append(nn.BatchNorm1d(dnh))
        layers.append(nn.LeakyReLU(.2))

        for i in range(hlayers-1):
            layers.append(nn.Linear(dnh, dnh))
            layers.append(nn.BatchNorm1d(dnh))
            layers.append(nn.LeakyReLU(.2))

        layers.append(nn.Linear(dnh, 1))  # 最終出力

        self.layers = nn.Sequential(*layers)

        # ----------------------------------
        # maskの用意（対角成分のみ1で、他は0の行列）
        # ----------------------------------
        mask = torch.eye(nfeatures, nfeatures)  # 変数の数×変数の数の単位行列
        self.register_buffer("mask", mask.unsqueeze(0))  # 単位行列maskを保存しておく

        # 注意：register_bufferはmodelのパラメータではないが、その後forwardで使う変数を登録するPyTorchのメソッドです
        # self.変数名で、以降も使用可能になります
        # https://pytorch.org/docs/stable/nn.html?highlight=register_buffer#torch.nn.Module.register_buffer

    def forward(self, input, obs_data=None):
        """　順伝搬の計算
        Args:
            input (torch.Size([データ数, 観測変数の種類数])): 観測したデータ、もしくは生成されたデータ
            obs_data (torch.Size([データ数, 観測変数の種類数])):観測したデータ
        Returns:
            torch.Tensor: 観測したデータか、それとも生成されたデータかの判定結果
        """

        if obs_data is not None:
            # 生成データを識別器に入力する場合
            return [self.layers(i) for i in torch.unbind(obs_data.unsqueeze(1) * (1 - self.mask)
                                                         + input.unsqueeze(1) * self.mask, 1)]
            # 対角成分のみ生成したデータ、その他は観測データに
            # データを各変数ごとに、生成したもの、その他観測したもので混ぜて、1変数ずつ生成したものを放り込む
            # torch.unbind(x,1)はxの1次元目でテンソルをタプルに展開する
            # minibatch数が2000、観測データの変数が6種類の場合、
            # [2000,6]→[2000,6,6]→([2000,6], [2000,6], [2000,6], [2000,6], [2000,6], [2000,6])→([2000,1], [2000,1], [2000,1], [2000,1], [2000,1], [2000,1])
            # returnは[torch.Size([2000, 1]),torch.Size([2000, 1]),torch.Size([2000, 1], torch.Size([2000, 1]),torch.Size([2000, 1]),torch.Size([2000, 1])]

            # 注：生成した変数全種類を用いた判定はしない。
            # すなわち、生成した変数1種類と、元の観測データたちをまとめて1つにし、それが観測結果か、生成結果を判定させる

        else:
            # 観測データを識別器に入力する場合

            return self.layers(input)
            # returnは[torch.Size([2000, 1])]

    def reset_parameters(self):
        """識別器Dの重みパラメータの初期化を実施"""
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class SAMGenerator(nn.Module):
    """SAMのGeneratorのニューラルネットワーク
    """

    def __init__(self, data_shape, nh):
        """初期化"""
        super(SAMGenerator, self).__init__()

        # ----------------------------------
        # 対角成分のみ0で、残りは1のmaskとなる変数skeletonを作成
        # ※最後の行は、全部1です
        # ----------------------------------
        nb_vars = data_shape[1]  # 変数の数
        skeleton = 1 - torch.eye(nb_vars + 1, nb_vars)

        self.register_buffer('skeleton', skeleton)

        # 注意：register_bufferはmodelのパラメータではないが、その後forwardで使う変数を登録するPyTorchのメソッドです
        # self.変数名で、以降も使用可能になります
        # https://pytorch.org/docs/stable/nn.html?highlight=register_buffer#torch.nn.Module.register_buffer

        # ----------------------------------
        # ネットワークの用意
        # ----------------------------------
        # 入力層（SAMの形での全結合層）　
        self.input_layer = Linear3D(
            (nb_vars, nb_vars + 1, nh))  # nhは中間層のニューロン数
        # https://github.com/FenTechSolutions/CausalDiscoveryToolbox/blob/32200779ab9b63762be3a24a2147cff09ba2bb72/cdt/utils/torch.py#L289

        # 中間層
        layers = []
        # 2次元を1次元に変換してバッチノーマライゼーションするモジュール
        layers.append(ChannelBatchNorm1d(nb_vars, nh))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

        # ChannelBatchNorm1d
        # https://github.com/FenTechSolutions/CausalDiscoveryToolbox/blob/32200779ab9b63762be3a24a2147cff09ba2bb72/cdt/utils/torch.py#L130

        # 出力層（再度、SAMの形での全結合層）
        self.output_layer = Linear3D((nb_vars, nh, 1))

    def forward(self, data, noise, adj_matrix, drawn_neurons=None):
        """　順伝搬の計算
        Args:
            data (torch.Tensor): 観測データ
            noise (torch.Tensor): データ生成用のノイズ
            adj_matrix (torch.Tensor): 因果関係を示す因果構造マトリクスM
            drawn_neurons (torch.Tensor): Linear3Dの複雑さを制御する複雑さマトリクスZ
        Returns:
            torch.Tensor: 生成されたデータ
        """

        # 入力層
        x = self.input_layer(data, noise, adj_matrix *
                             self.skeleton)  # Linear3D

        # 中間層（バッチノーマライゼーションとTanh）
        x = self.layers(x)

        # 出力層
        output = self.output_layer(
            x, noise=None, adj_matrix=drawn_neurons)  # Linear3D

        return output.squeeze(2)

    def reset_parameters(self):
        """重みパラメータの初期化を実施"""

        self.input_layer.reset_parameters()
        self.output_layer.reset_parameters()

        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


def notears_constr(adj_m, max_pow=None):
    """No Tears constraint for binary adjacency matrixes.
    Args:
        adj_m (array-like): Adjacency matrix of the graph
        max_pow (int): maximum value to which the infinite sum is to be computed.
           defaults to the shape of the adjacency_matrix
    Returns:
        np.ndarray or torch.Tensor: Scalar value of the loss with the type
            depending on the input.
    参考：https://github.com/FenTechSolutions/CausalDiscoveryToolbox/blob/32200779ab9b63762be3a24a2147cff09ba2bb72/cdt/utils/loss.py#L215
    """
    m_exp = [adj_m]
    if max_pow is None:
        max_pow = adj_m.shape[1]
    while(m_exp[-1].sum() > 0 and len(m_exp) < max_pow):
        m_exp.append(m_exp[-1] @ adj_m/len(m_exp))

    return sum([i.diag().sum() for idx, i in enumerate(m_exp)])


def run_SAM(in_data, lr_gen, lr_disc, lambda1, lambda2, hlayers, nh, dnh, train_epochs, test_epochs, device):
    '''SAMの学習を実行する関数'''

    # ---------------------------------------------------
    # 入力データの前処理
    # ---------------------------------------------------
    list_nodes = list(in_data.columns)  # 入力データの列名のリスト
    data = scale(in_data[list_nodes].values)  # 入力データの正規化
    nb_var = len(list_nodes)  # 入力データの数 = d
    data = data.astype('float32')  # 入力データをfloat32型に
    data = torch.from_numpy(data).to(device)  # 入力データをPyTorchのテンソルに
    rows, cols = data.size()  # rowsはデータ数、colsは変数の数

    # ---------------------------------------------------
    # DataLoaderの作成（バッチサイズは全データ）
    # ---------------------------------------------------
    batch_size = rows  # 入力データ全てを使用したミニバッチ学習とする
    data_iterator = DataLoader(data, batch_size=batch_size,
                               shuffle=True, drop_last=True)
    # 注意：引数のdrop_lastはdataをbatch_sizeで取り出していったときに最後に余ったものは使用しない設定

    # ---------------------------------------------------
    # 【Generator】ネットワークの生成とパラメータの初期化
    # cols：入力変数の数、nhは中間ニューロンの数、hlayersは中間層の数
    # neuron_samplerは、Functional gatesの変数zを学習するネットワーク
    # graph_samplerは、Structual gatesの変数aを学習するネットワーク
    # ---------------------------------------------------
    sam = SAMGenerator((batch_size, cols), nh).to(device)  # 生成器G
    graph_sampler = MatrixSampler(nb_var, mask=None, gumbel=False).to(
        device)  # 因果構造マトリクスMを作るネットワーク
    neuron_sampler = MatrixSampler((nh, nb_var), mask=False, gumbel=True).to(
        device)  # 複雑さマトリクスZを作るネットワーク

    # 注意：MatrixSamplerはGumbel-Softmaxを使用し、0か1を出力させるニューラルネットワーク
    # SAMの著者らの実装モジュール、MatrixSamplerを使用
    # https://github.com/FenTechSolutions/CausalDiscoveryToolbox/blob/32200779ab9b63762be3a24a2147cff09ba2bb72/cdt/utils/torch.py#L212

    # 重みパラメータの初期化
    sam.reset_parameters()
    graph_sampler.weights.data.fill_(2)

    # ---------------------------------------------------
    # 【Discriminator】ネットワークの生成とパラメータの初期化
    # cols：入力変数の数、dnhは中間ニューロンの数、hlayersは中間層の数。
    # ---------------------------------------------------
    discriminator = SAMDiscriminator(cols, dnh, hlayers).to(device)
    discriminator.reset_parameters()  # 重みパラメータの初期化

    # ---------------------------------------------------
    # 最適化の設定
    # ---------------------------------------------------
    # 生成器

    g_optimizer = optim.Adam(sam.parameters(), lr=lr_gen)
    graph_optimizer = optim.Adam(graph_sampler.parameters(), lr=lr_gen)
    neuron_optimizer = optim.Adam(neuron_sampler.parameters(), lr=lr_gen)

    # 識別器
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr_disc)

    # 損失関数
    criterion = nn.BCEWithLogitsLoss()
    # nn.BCEWithLogitsLoss()は、binary cross entropy with Logistic function
    # https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss

    # 損失関数のDAGに関する制約の設定パラメータ
    dagstart = 0.5
    dagpenalization_increase = 0.001*10

    # ---------------------------------------------------
    # forward計算、および損失関数の計算に使用する変数を用意
    # ---------------------------------------------------
    _true = torch.ones(1).to(device)
    _false = torch.zeros(1).to(device)

    noise = torch.randn(batch_size, nb_var).to(device)  # 生成器Gで使用する生成ノイズ
    noise_row = torch.ones(1, nb_var).to(device)

    output = torch.zeros(nb_var, nb_var).to(device)  # 求まった隣接行列
    output_loss = torch.zeros(1, 1).to(device)

    # ---------------------------------------------------
    # forwardの計算で、ネットワークを学習させる
    # ---------------------------------------------------
    pbar = tqdm(range(train_epochs + test_epochs))  # 進捗（progressive bar）の表示

    for epoch in pbar:
        for i_batch, batch in enumerate(data_iterator):

            # 最適化を初期化
            g_optimizer.zero_grad()
            graph_optimizer.zero_grad()
            neuron_optimizer.zero_grad()
            d_optimizer.zero_grad()

            # 因果構造マトリクスM（drawn_graph）と複雑さマトリクスZ（drawn_neurons）をMatrixSamplerから取得
            drawn_graph = graph_sampler()
            drawn_neurons = neuron_sampler()
            # (drawn_graph)のサイズは、torch.Size([nb_var, nb_var])。 出力値は0か1
            # (drawn_neurons)のサイズは、torch.Size([nh, nb_var])。 出力値は0か1

            # ノイズをリセットし、生成器Gで疑似データを生成
            noise.normal_()
            generated_variables = sam(data=batch, noise=noise,
                                      adj_matrix=torch.cat(
                                          [drawn_graph, noise_row], 0),
                                      drawn_neurons=drawn_neurons)

            # 識別器Dで判定
            # 観測変数のリスト[]で、各torch.Size([data数, 1])が求まる
            disc_vars_d = discriminator(generated_variables.detach(), batch)
            # 観測変数のリスト[] で、各torch.Size([data数, 1])が求まる
            disc_vars_g = discriminator(generated_variables, batch)
            true_vars_disc = discriminator(batch)  # torch.Size([data数, 1])が求まる

            # 損失関数の計算（DCGAN）
            disc_loss = sum([criterion(gen, _false.expand_as(gen)) for gen in disc_vars_d]) / nb_var \
                + criterion(true_vars_disc, _true.expand_as(true_vars_disc))

            gen_loss = sum([criterion(gen,
                                      _true.expand_as(gen))
                            for gen in disc_vars_g])

            # 損失の計算（SAM論文のオリジナルのfgan）
            #disc_loss = sum([torch.mean(torch.exp(gen - 1)) for gen in disc_vars_d]) / nb_var - torch.mean(true_vars_disc)
            #gen_loss = -sum([torch.mean(torch.exp(gen - 1)) for gen in disc_vars_g])

            # 識別器Dのバックプロパゲーションとパラメータの更新
            if epoch < train_epochs:
                disc_loss.backward()
                d_optimizer.step()

            # 生成器のGの損失の計算の残り（マトリクスの複雑さとDAGのNO TEAR）
            struc_loss = lambda1 / batch_size*drawn_graph.sum()     # Mのloss
            func_loss = lambda2 / batch_size*drawn_neurons.sum()   # Aのloss

            regul_loss = struc_loss + func_loss

            if epoch <= train_epochs * dagstart:
                # epochが基準前のときは、DAGになるようにMへのNO TEARSの制限はかけない
                loss = gen_loss + regul_loss

            else:
                # epochが基準後のときは、DAGになるようにNO TEARSの制限をかける
                filters = graph_sampler.get_proba()  # マトリクスMの要素を取得（ただし、0,1ではなく、1の確率）
                dag_constraint = notears_constr(filters*filters)  # NO TERARの計算

                # 徐々に線形にDAGの正則を強くする
                loss = gen_loss + regul_loss + \
                    ((epoch - train_epochs * dagstart) *
                     dagpenalization_increase) * dag_constraint

            if epoch >= train_epochs:
                # testのepochの場合、結果を取得
                output.add_(filters.data)
                output_loss.add_(gen_loss.data)
            else:
                # trainのepochの場合、生成器Gのバックプロパゲーションと更新
                # retain_graph=Trueにすることで、以降3つのstep()が実行できる
                loss.backward(retain_graph=True)
                g_optimizer.step()
                graph_optimizer.step()
                neuron_optimizer.step()

            # 進捗の表示
            if epoch % 50 == 0:
                pbar.set_postfix(gen=gen_loss.item()/cols,
                                 disc=disc_loss.item(),
                                 regul_loss=regul_loss.item(),
                                 tot=loss.item())

    return output.cpu().numpy()/test_epochs, output_loss.cpu().numpy()/test_epochs/cols  # Mと損失を出力

# --------------------------------
# Copyright (c) 2020 Daichi Furukawa


class SAM:
    def __init__(self, lr_gen=0.01 * 0.5, lr_disc=0.01 * 0.5 * 2,
                 lambda1=0.01, lambda2=0.005 * 20,
                 hlayers=2, nh=200, dnh=200,
                 train_epochs=10000, test_epochs=1000,
                 device='cuda:0'):
        """Init and parametrize the SAM model"""
        super(SAM, self).__init__()
        self.lr_gen = lr_gen
        self.lr_disc = lr_disc
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.hlayers = hlayers
        self.nh = nh
        self.dnh = dnh
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.device = device

    def predict(self, in_data, run_times=16):
        m_list = []
        loss_list = []

        for i in range(run_times):
            print(f"running {i+1}/{run_times} ...")
            m, loss = run_SAM(in_data=in_data, lr_gen=self.lr_gen,
                              lr_disc=self.lr_disc,
                              # lambda1=0.01, lambda2=1e-05,
                              lambda1=self.lambda1, lambda2=self.lambda2,
                              hlayers=self.hlayers,
                              nh=self.nh, dnh=self.dnh,
                              train_epochs=self.train_epochs,
                              test_epochs=self.test_epochs,
                              device='cuda:0')

            print(loss)
            print(m)

            m_list.append(m)
            loss_list.append(loss)

        # ネットワーク構造（平均）
        return sum(m_list) / len(m_list)
