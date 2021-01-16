import pandas as pd
import numpy as np
import pyper
from graphviz import Source
import webbrowser
from os import getcwd, rename, remove as rm_file

from SAM import SAM
from testdata import make_test_data

np.set_printoptions(precision=2, floatmode='fixed', suppress=True)
np.random.seed(0)

""" --- make pyper instance ---
"""
r = pyper.R(use_pandas='True')

""" --- import csv file ---
"""
print("current: " + getcwd())
# filename = input("csv file name? >> ")
# df = pd.read_csv(filename)
df = make_test_data(size=2000)

cols = list(df.columns)

""" --- data pre-processing ---
"""

""" --- run SAM ---
"""
# セルフ推論
# causal_discovered_data = np.array([[0, 0, 0, 1],
#                                    [0, 0, 0, 1],
#                                    [1, 1, 0, 1],
#                                    [0, 0, 0, 0]])

causal_discovered_data = SAM(train_epochs=10000, test_epochs=1000).predict(in_data=df, run_times=16)
# テスト用
# causal_discovered_data = np.array([[0, 0, 0, 0],
#                                    [1, 0, 0, 0],
#                                    [1, 0, 0, 0],
#                                    [1, 1, 1, 0]])
# 転置（すなわち因果関係を逆方向に）したら適合度上がることも
# causal_discovered_data = causal_discovered_data.transpose()

print(pd.DataFrame(data=causal_discovered_data, index=cols, columns=cols))
threshold = float(input("threshold? >> "))
causal_discovered_data = causal_discovered_data >= threshold

""" ---  forward dataframe to R as "data" ---
"""
r.assign("data", df)

""" --- calc corr ---
"""
r("corr <- cor(data)")

""" --- import libraries used in R ---
"""
r("library(sem)")
r("library(DiagrammeR)")

""" --- make SEM model ---
"""
model_text = "model <- specifyModel(text=\"\n"
causal_index = list(zip(*np.where(causal_discovered_data)))
counter = 1
for a, b in causal_index:
    if causal_discovered_data[b, a]:
        if a < b:
            model_text += f"{cols[a]} <-> {cols[b]},b{counter},NA\n"
    else:
        model_text += f"{cols[a]} -> {cols[b]},b{counter},NA\n"
    counter += 1
counter = 1
for val in cols:
    model_text += f"{val} <-> {val},e{counter},NA\n"
    counter += 1
model_text += "\")"

print(model_text)
r(model_text)

r("ans <- sem(model, corr, nrow(data))")
# print(r('stdCoef(ans)'))
print(r('summary(ans,fit.indices = c("GFI","AGFI","SRMR","RMSEA"))'))

""" --- rendering ---
"""
r('pathDiagram(ans, "output_diagram", output.type = "graphics",\
    ignore.double = FALSE, edge.labels = "values", digits = 3)')

output_format = input("output format? (png or svg or pdf) >> ")
output_filename = input("output file name? >> ") + '.' + output_format
Source.from_file(filename="./output_diagram.dot", format=output_format, engine='dot').render()
rename("output_diagram.dot." + output_format, output_filename)

""" --- remove temporary file ---
"""
try:
    rm_file("./output_diagram.dot")
    rm_file("./output_diagram.pdf")
except FileNotFoundError as e:
    pass

""" --- show rendered Diagram in web browser ---
"""
# Chrome指定
browser = webbrowser.get(
    '"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe" %s')
browser.open("file:///" + getcwd() + "/" + output_filename)
