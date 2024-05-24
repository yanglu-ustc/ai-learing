import torch
import numpy as np
import copy

from scipy.io import wavfile

from 作业.final_for_model.code.model_for_practice import Get_k, get_output, Net1, Net2, Net3, WeightedBCELoss, \
    get_index
from evalute import evaluate


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


class Practice:
    def __init__(self):
        self.net1 = None
        self.net2 = None
        self.net3 = None
        torch.set_default_dtype(torch.float64)

    def get_data(self, path):
        """
        获取csv数据，将多组音频的数据拼接
        :param path: 路径所组成的列表
        :return: 总的数据列表以及获取的每一段音频的长度，方便后续进行处理
        """
        k = 0
        len_arr = []
        for path_item in path:
            signal, signal_len, fs = read_wav(path_item)
            assert fs == 8000, "抱歉，请将频率转换为8000"
            data_ = np.abs(signal)
            len_arr.append(signal_len)
            if k == 0:
                the_data = data_
                k = 1
            else:
                the_data = np.concatenate((the_data, data_), axis=0)
        return the_data, len_arr

    def getOutput(self, len_Array, path):
        """
        获取人工打点的区间片段，并将其转换为要进行训练的矩阵
        :param len_Array: 每一个音频的长度的列表
        :param path: 路径所组成的列表
        :return: 矩阵
        """
        lenOfArray = len(len_Array)
        k = 0
        for i in range(lenOfArray):
            y_ = get_output(path[i], len_Array[i])
            if k == 0:
                combined_tensor = y_
                k = 1
            else:
                combined_tensor = torch.cat((combined_tensor, y_), dim=0)
        return combined_tensor

    def net_train(self, x1, x2, y):

        lr, num_epochs = 0.01, 1000

        self.net1 = Net1(n_feature=1)
        self.net2 = Net2(n_feature=1)
        self.net3 = Net3()

        criterion = WeightedBCELoss()
        optimizer = torch.optim.Adam(
            list(self.net1.parameters()) + list(self.net2.parameters()) + list(self.net3.parameters()), lr=lr)

        losses = []

        for epoch in range(num_epochs):  # 使用epoch作为迭代变量
            self.net1.train()
            self.net2.train()
            self.net3.train()
            # 数据集传入网络前向计算
            y1 = self.net1(x1)
            y2 = self.net2(x2)
            y_hat = self.net3(y1, y2)
            # 计算loss
            loss = criterion(y_hat, y)
            # 清除网络状态
            optimizer.zero_grad()
            # loss反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 输出loss大小
            losses.append(loss.item())

        list_label_data = []

        for name, param in list(self.net1.named_parameters()) + list(self.net2.named_parameters()) + list(
                self.net3.named_parameters()):
            print(name, param.data)
            list_label_data.append(param.data)

        print("\n")

        return list_label_data

    @staticmethod
    def get_index(y_hat):
        current_list = []
        all_lists = []

        prev_true_index = -1
        for index, value in enumerate(y_hat):
            if value:
                if prev_true_index == -1:
                    current_list.append(index)
                else:
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

        index = get_index(all_lists)
        return index

    def test(self, path_test):
        signal, signal_len, fs = read_wav(path_test)
        assert fs == 8000, "抱歉，请将频率转换为8000"
        data_ = np.abs(signal)

        the_k_data = Get_k(data_)
        k_ = the_k_data.get_k()
        k_data_ = torch.tensor(k_)
        k_data_ = torch.abs(k_data_)

        x1 = k_data_.reshape(-1, 1).to(dtype=torch.float64)
        x2 = torch.tensor(data_).reshape(-1, 1).to(dtype=torch.float64)

        self.net1.eval()
        self.net2.eval()
        self.net3.eval()
        y1 = self.net1(x1)
        y2 = self.net2(x2)
        y_hat = self.net3(y1, y2)
        y_hat_ = y_hat[:] > 0.75

        all_list = Practice.get_index(y_hat=y_hat_)

        return all_list, signal_len


def train_net(practice, path_train, path_out):
    data, len_array = practice.get_data(path_train)

    the_k_data = Get_k(data)
    k_ = the_k_data.get_k()
    k_data_ = torch.tensor(k_)
    k_data_ = torch.abs(k_data_)

    x1 = k_data_.reshape(-1, 1).to(dtype=torch.float64)
    x2 = torch.tensor(data).reshape(-1, 1).to(dtype=torch.float64)

    y = practice.getOutput(len_array, path_out)

    list_label = practice.net_train(x1, x2, y)

    return list_label


def train_k(practice, path_train, path_out):
    """
    1折交叉验证
    :param practice: 训练对象
    :param path_train: 训练集
    :param path_out: 标记的结果
    :return:
    """
    len_files = len(path_out)
    assert len_files == len(path_train), '文件传参错误！！！！！'
    list_label_datas = []
    for i in range(len_files):
        path_train_ = copy.deepcopy(path_train)
        path_out_ = copy.deepcopy(path_out)
        path_train_.remove(path_train_[i])
        path_out_.remove(path_out_[i])
        list_label = train_net(practice, path_train_, path_out_)

        predict_data, len_data = practice.test(path_train[i])
        for _ in predict_data:
            print(_[0], _[1], sep=",")
        label_data = read_txt(path_out[i])

        f1_score, accuracy, recall, precision = evaluate(len_data, label_data, predict_data)
        print('\n')
        print('f1_score: ', f1_score)
        print('accuracy: ', accuracy)
        print('recall: ', recall)
        print('precision: ', precision)
        print('\n')

        list_label_datas.append(list_label)

    return list_label_datas


if __name__ == '__main__':
    practice = Practice()

    list_label_datas = train_k(practice, ['../data/data_1.wav', '../data/data_5.wav', '../data/data_10.wav'],
                               ['../data/label/data_1.txt', '../data/label/data_5.txt', '../data/label/data_10.txt'])

    list_label_datas = np.array(list_label_datas)
    list_label_datas = np.average(list_label_datas, axis=0)
    print(list_label_datas)
