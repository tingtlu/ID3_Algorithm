import math

#train_data训练数据集，_attributes数据中的各中属性集
class ID3():
    def __init__(self, train_data):
        self.data = train_data
        self.target_attribute = train_data.columns[-1]
        self._attributes = train_data.columns[:-1]

    #获取最大增益熵对应的属性
    def get_max_entropy_attr(self):
        max_entropy = 0
        for attribute in self._attributes:
            gain_entropy = self.gain(self.data, attribute)
            if gain_entropy > max_entropy:
                max_entropy = gain_entropy
                max_entropy_attr = attribute
        if max_entropy == 0:
            return self.target_attribute

        return max_entropy_attr

    #计算每种属性信息的增益熵
    def gain(self, _data, attr_value):
        # 计算供学习的各个属性熵
        attr_ent = 0
        for value in set(self.data[attr_value]):
            sub_data = self.data[self.data[attr_value] == value]
            attr_ent = attr_ent + (sub_data.__len__() / self.data.__len__()) * self.entropy(sub_data)

        return self.entropy(_data) - attr_ent


    #计算数据集关于类标记的信息熵
    def entropy(self, dataset):
        entropy = 0
        for value in set(dataset[self.target_attribute]):
            sub_set = dataset[dataset[self.target_attribute] == value]
            p = sub_set.__len__() / dataset.__len__()
            entropy = entropy + (-p) * math.log(p, 2)

        return entropy



