import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Net1(torch.nn.Module):
    # 初始化函数，接受自定义输入特征维数，隐藏层特征维数，输出层特征维数
    def __init__(self, n_feature, alpha_init=50):
        super(Net1, self).__init__()
        self.predict = torch.nn.Linear(n_feature, 1, dtype=torch.float64)
        nn.init.constant_(self.predict.bias, -alpha_init)
        nn.init.constant_(self.predict.weight, 1)
        for name, param in self.predict.named_parameters():
            if 'weight' in name:
                param.requires_grad = False

    # 前向传播过程
    def forward(self, x):
        x = self.predict(x)
        x = F.relu(x)
        return x


class Net2(torch.nn.Module):
    # 初始化函数，接受自定义输入特征维数，隐藏层特征维数，输出层特征维数以及alpha的初始值
    def __init__(self, n_feature, alpha_init=10000):
        super(Net2, self).__init__()
        self.predict = torch.nn.Linear(n_feature, 1, dtype=torch.float64)
        nn.init.constant_(self.predict.bias, -alpha_init)
        nn.init.constant_(self.predict.weight, 1)
        for name, param in self.predict.named_parameters():
            if 'weight' in name:
                param.requires_grad = False

    # 前向传播过程
    def forward(self, x):
        x = self.predict(x)
        x = F.relu(x)
        return x


class Net3(nn.Module):
    def __init__(self, n_feature=2, alpha_init=0.0):
        super(Net3, self).__init__()
        self.predict = torch.nn.Linear(2, 1, dtype=torch.float64)

    def forward(self, y1, y2):
        x = torch.cat((y1, y2), dim=1)
        x = self.predict(x)
        out = torch.sigmoid(x)
        return out


class Get_k:
    def __init__(self, data, left_avg=0, right_avg=0, length=100):
        self.data = list(data)
        self.left_avg = left_avg
        self.right_avg = right_avg
        self.length = length

    def add_to_data(self):
        left_list = []
        for i in range(self.length):
            left_list.append(self.left_avg)
        right_list = []
        for i in range(self.length):
            right_list.append(self.right_avg)
        data = left_list
        data.extend(self.data)
        data.extend(right_list)
        return data

    def get_k(self):
        list_k_1 = []
        data = self.add_to_data()
        for i in range(self.length, len(data) - self.length):
            k_1 = (data[i + self.length] - data[i]) / self.length
            k_2 = (-data[i - self.length] + data[i]) / self.length
            k_3 = (data[i + self.length] - data[i - self.length]) / (self.length * 2)
            k_ = (k_1 + k_2 + k_3) / 3
            list_k_1.append(k_)
        return list_k_1


def get_output(path, len_, delimiter=','):
    output = np.genfromtxt(path, delimiter=delimiter)
    # 获取结果 y
    col = torch.zeros([1, len_])
    for data_ in output:
        num_left = int(data_[0])
        num_right = int(data_[1])
        col[0][num_left:num_right] = 1
    y = col
    return y.reshape(1, -1).T


class WeightedBCELoss(torch.nn.Module):
    def __init__(self, weight_for_one=10.0):
        super(WeightedBCELoss, self).__init__()
        self.weight_for_one = weight_for_one

    def forward(self, inputs, targets):
        # 确保inputs和targets具有相同的形状
        assert inputs.size() == targets.size(), "Input and target tensors should have the same shape"

        # 计算正常的BCELoss
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 找到预测为1但真实值为0的样本（即需要增大权重的样本）
        wrong_ones = (inputs < 0.5) & (targets == 1)

        # 仅对这些样本的损失应用更大的权重
        loss[wrong_ones] *= self.weight_for_one

        # 对损失进行平均或求和（取决于你的需求）
        return loss.mean()


def get_index(data):
    list_output = []
    for one_list in data:
        current_list = []
        current_list.append(one_list[0])
        current_list.append(one_list[-1])
        list_output.append(current_list)
    return list_output
