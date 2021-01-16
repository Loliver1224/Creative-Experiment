import pandas as pd
import numpy as np
from numpy.random import randn, choice
from scipy.special import expit

np.random.seed(65535)


def make_test_data(size=2000):

    # 大きさ
    x1 = choice([0, 1, 2], size=size, p=[0.3, 0.3, 0.4])

    # 見やすさ
    e_x2 = expit(randn(size))  # ノイズ
    x2_prob = 0.5
    x2 = x2_prob * x1 + (1 - x2_prob) * (e_x2 * 2)
    x2 = np.round(x2).astype(int)

    # 押しやすさ
    e_x3 = expit(randn(size))  # ノイズ
    x3_prob = 0.4
    x3 = x3_prob * x1 + (1 - x3_prob) * (e_x3 * 2)
    x3 = np.round(x3).astype(int)

    # 満足度
    Y = (x1 * 0.3 + x2 * 0.4 + x3 * 0.3) * (6 / 2)
    Y = Y.astype(int)

    # 1,2,3 | 1,2,...7 スケールへ
    x1 += 1
    x2 += 1
    x3 += 1
    Y += 1

    df = pd.DataFrame({'easy2view': x2,
                       'easy2push': x3,
                       'size': x1,
                       'satisfaction': Y
                       })
    print(df.head())
    return df


if __name__ == '__main__':
    make_test_data()
