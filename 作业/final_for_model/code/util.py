import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from 作业.final_for_model.code.evalute import evaluate
from 作业.final_for_model.code.model_for_practice import Get_k, get_index

torch.set_default_dtype(torch.float64)


def read_wav(fname):
    fs, signal = wavfile.read(fname)
    if len(signal.shape) != 1:
        print("convert stereo to mono")
        signal = signal[:, 0]
    signal = signal.flatten()
    signal_len = len(signal)
    return signal, signal_len, fs


def read_txt(file_dir):
    data = []

    # read txt
    with open(file_dir, 'r') as file:
        lines = file.readlines()
        # process data
        for line in lines:
            # read data
            values = [int(value) for value in line.strip().split(",")]
            data.append(values)

    # set data format as numpy
    data = np.array(data)

    return data


class Net(torch.nn.Module):
    def __init__(self, n_feature_1, n_feature_2, net_1_weight, net_1_bias, net_2_weight, net_2_bias, net_3_weight,
                 net_3_bias):
        super(Net, self).__init__()

        self.predict_1 = torch.nn.Linear(n_feature_1, 1, dtype=torch.float64)
        nn.init.constant_(self.predict_1.bias, net_1_bias)
        nn.init.constant_(self.predict_1.weight, net_1_weight)

        self.predict_2 = torch.nn.Linear(n_feature_2, 1, dtype=torch.float64)
        nn.init.constant_(self.predict_2.bias, net_2_bias)
        nn.init.constant_(self.predict_2.weight, net_2_weight)

        self.predict_3 = torch.nn.Linear(2, 1, dtype=torch.float64)
        nn.init.constant_(self.predict_3.bias, net_3_bias)
        self.predict_3.weight.data = net_3_weight

    # 前向传播过程
    def forward(self, x_1, x_2):
        x_1 = self.predict_1(x_1)
        x_1 = F.relu(x_1)
        x_2 = self.predict_2(x_2)
        x_2 = F.relu(x_2)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.predict_3(x)
        out = torch.sigmoid(x)
        return out


def final_test(net_param, path, path_output, path_label=None):
    """
    :param path_label: 人工标记的区间，可以不进行赋值，赋值时将计算精确度等值
    :param path_output: 要进行预测的区间的存储位置的路径
    :param net_param: 网络的信息
    :param path: 音频信息的文件的路径
    :return:
    """
    n_feature_1 = 1
    n_feature_2 = 1
    net_1_weight = net_param[0]
    net_1_bias = net_param[1]
    net_2_weight = net_param[2]
    net_2_bias = net_param[3]
    net_3_weight = torch.tensor(net_param[4]).reshape(1, -1)
    net_3_bias = net_param[5]

    net = Net(n_feature_1, n_feature_2, net_1_weight, net_1_bias, net_2_weight, net_2_bias, net_3_weight,
              net_3_bias)

    signal, signal_len, fs = read_wav(path)
    assert fs == 8000, "抱歉，请将频率转换为8000"
    x_1 = np.abs(signal)

    the_k_data = Get_k(x_1)
    k_ = the_k_data.get_k()
    k_data_ = torch.tensor(k_)
    k_data_ = torch.abs(k_data_)

    x1 = k_data_.reshape(1, -1).to(dtype=torch.float64)
    x2 = torch.tensor(x_1).reshape(1, -1).to(dtype=torch.float64)
    x1 = x1.T
    x2 = x2.T

    # 测试效果的分析
    net.eval()
    # 数据集传入网络前向计算
    y_hat = net(x1, x2)
    y_hat_ = torch.zeros(y_hat.size(0), dtype=torch.bool)
    y_hat_ = y_hat[:] > 0.75

    current_list = []
    # 初始化一个大的列表来存储所有小列表
    all_lists = []

    # 初始化索引和前一个True的index
    prev_true_index = -1
    for index, value in enumerate(y_hat_):
        if value:
            # 如果是第一个True
            if prev_true_index == -1:
                current_list.append(index)
            else:
                # 判断差值是否大于100
                diff = index - prev_true_index
                if diff <= 2400:
                    current_list.append(index)
                else:
                    if current_list[-1] - current_list[0] > 800:
                        all_lists.append(current_list)
                    current_list = [index]
            prev_true_index = index

    if current_list and current_list[-1] - current_list[0] > 800:
        all_lists.append(current_list)

    predict_ = get_index(all_lists)
    predict = np.array(predict_)
    for index_ in predict:
        print(index_[0], index_[1], sep=",")

    # 打开一个文件用于写入，使用 'w' 模式（如果文件已存在，它将被覆盖）
    with open(path_output, 'w') as file:
        # 遍历列表中的每个子列表
        for sublist in predict_:
            if sublist:
                line = ','.join(map(str, sublist))
                file.write(line + '\n')
            else:
                pass
    if path_label is None:
        pass
    else:
        label_data = read_txt(path_label)
        f1_score, accuracy, recall, precision = evaluate(signal_len, label_data, predict)
        print('\n')
        print('f1_score: ', f1_score)
        print('accuracy: ', accuracy)
        print('recall: ', recall)
        print('precision: ', precision)
        print('\n')


final_test([1., -23.6830, 1., -19997.8765, [1.5892, 2.1599], -0.0006],
           '../data/data_1.wav', '../data/predict/data_1.txt', path_label='../data/label/data_1.txt')

final_test([1., -23.6830, 1., -19997.8765, [1.5892, 2.1599], -0.0006],
           '../data/data_5.wav', '../data/predict/data_5.txt', path_label='../data/label/data_5.txt')

final_test([1., -23.6830, 1., -19997.8765, [1.5892, 2.1599], -0.0006],
           '../data/data_10.wav', '../data/predict/data_10.txt', path_label='../data/label/data_10.txt')

# WAV_PATH：wav文件路径
# PREDICT_PATH：要进行预测的区间的存储位置的路径
# LABEL_PATH：人工标记的区间，可以不进行赋值，赋值时将计算精确度等值
# final_test([1., -23.6830, 1., -19997.8765, [1.5892, 2.1599], -0.0006],
#            WAV_PATH, PREDICT_PATH, path_label=LABEL_PATH)
