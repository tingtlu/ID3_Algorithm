from ID3 import *


#根据传入的数据集，利用ID3算法，创建决策树集合
def create_id3_tree(dataset):
    id3 = ID3(dataset)
    attr = id3.get_max_entropy_attr()
    feature = dataset.columns[-1]

    #终止条件
    if attr == feature or len(set(dataset[feature])) == 1:
        return list(set(dataset[feature]))[0]

    tree = {}
    tree[attr] = {}

    #迭代递归调用个子树
    for value in set(dataset[attr]):
        sub_data = dataset[dataset[attr] == value]
        sub_tree = create_id3_tree(sub_data)
        tree[attr][value] = sub_tree

    return tree


