import pandas as pd
import create_tree as ct
import tree_plot as tp

#读取数据集文件
data = pd.read_csv('data/dataset.csv')

#传入数据集，生成ID3决策树
tree = ct.create_id3_tree(data)

#画出决策树
tp.createPlot(tree)