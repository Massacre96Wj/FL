# -*- coding: UTF-8 -*-
"""
@Author ：WangJie
@Date   ：2020/11/29  10:10
@Desc   ：
"""
import datetime
import os
import time

import torch
from scipy.interpolate import make_interp_spline
from statsmodels.tsa.seasonal import seasonal_decompose
from torch import nn
import numpy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from functools import reduce
from log import logger
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

look_back = 4
EPOCH = 10
SIZE = 150
dataset = None
Nums = 0

# 提取特征函数
def create_dataset(dataset, look_back=4):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

#相对损失
def evluation(real, pred):
    return numpy.abs(real-pred)/real

scaler = MinMaxScaler(feature_range=(0, 1))

# def load_data(path, flag=False):
#     dataframe = read_csv(path, usecols=[0], engine='python', skipfooter=3)
#     global dataset
#     dataset = dataframe.values
#
#     # logger.info(type(dataset))    # numpy.array
#     # 将整型变为float
#     dataset = dataset.astype('float32')
#     # fix random seed for reproducibility
#     numpy.random.seed(7)
#     # normalize the dataset
#     dataset = scaler.fit_transform(dataset)
#     data_X, data_Y = create_dataset(dataset, look_back)
#     if flag:
#         return data_X, data_Y
#     else:
#         return data_X

# 初始化数据
dataset = numpy.zeros((30, SIZE), dtype=float)
d1 = numpy.random.poisson(lam=20.0, size=SIZE)
dataset[1] = d1
dataset[2] = numpy.random.poisson(lam=20.0, size=SIZE)
dataset[4] = numpy.random.poisson(lam=20.0, size=SIZE)

# 返回包含三个部分 trend（趋势部分），seasonal（季节性部分）和residual (残留部分)
def decompose(timeseries):
    decomposition = seasonal_decompose(timeseries, freq=3)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    return trend, seasonal, residual

def get_data(node_id, time_series):
    time_series = numpy.array(time_series)
    time_series = pd.Series(numpy.log(time_series + 1).reshape(-1), index=pd.DatetimeIndex(range(len(time_series))))
    trend, seasonal, residual = decompose(time_series)
    ts = trend + seasonal
    ts = ts.fillna(numpy.nanmean(numpy.array(ts)))
    dataset[node_id - 1] = ts.values

# 判断参与的节点有哪些
def join_train(dataset=dataset):
    global Nums
    cur_dataset = []
    cur_node = []
    for i, data in enumerate(dataset):
        if not (data == 0).all():
            cur_dataset.append(data.tolist())
            cur_node.append(i)
    Nums = len(cur_node)
    return numpy.array(cur_dataset), cur_node,

# 创建数据
def load_data1(dataset):
    # dataset = numpy.random.poisson(lam=lambd, size=300).reshape(-1, 1)
    # 将整型变为float
    dataset = dataset.astype('float32')
    # normalize the dataset
    dataset = scaler.fit_transform(dataset)
    data_X, data_Y = create_dataset(dataset, look_back)
    return data_X, data_Y

class LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=Nums, num_layer=1):
        super(LSTM, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        x = torch.relu(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x


class TrainPredict(object):

    def __init__(self):
        self.var_x, self.var_y, self.model = None, None, None
        self.Node = []

    def transfrom(self):
        global dataset
        dataset, self.Node = join_train()
        dfs_x, dfs_y = [], []
        for data in dataset:
            data_X, data_Y = load_data1(data.reshape(-1, 1))
            dfs_x.append(data_X)
            dfs_y.append(data_Y.reshape(-1, 1))
        dataset = dataset = numpy.zeros((30, SIZE), dtype=float)
        data_X = reduce(lambda left, right: numpy.concatenate((left, right), axis=1), dfs_x)
        data_Y = reduce(lambda left, right: numpy.concatenate((left, right), axis=1), dfs_y)
        train_X = data_X
        train_Y = data_Y
        train_X = numpy.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
        train_Y = train_Y.reshape(-1, 1, train_Y.shape[1])
        self.var_x = torch.from_numpy(train_X)
        self.var_y = torch.from_numpy(train_Y)
        return train_X.shape[-1]
        # print(self.model)
        # print(dataset)

    def train(self):
        input_shape = self.transfrom()

        self.model = LSTM(input_shape, Nums*look_back*4, Nums, 1)
        loss_fun = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)

        writer = SummaryWriter("logs")
        writer.add_graph(self.model, input_to_model=torch.zeros(1, 1, Nums*look_back), verbose=True)
        loss_plot = [1]
        time_plot = [0]
        first = 0
        for epoch in range(1, EPOCH+1):
            logger.info("Epoch: %d......" % epoch)
            if not first:
                first = float(time.time())
                writer.add_scalar("train loss", 1.0, 0)
            for t in range(25):
                # 前向传播
                out = self.model(self.var_x)
                loss = loss_fun(out, self.var_y)
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # tensorboard --logdir=logs
            writer.add_scalar("train loss", loss.item(), float(time.time()*1000) - first*1000)
            time_plot.append(float(time.time()) - first)
            loss_plot.append(loss.item())

        x_smooth = numpy.linspace(min(time_plot), max(time_plot), 300)
        y_smooth = make_interp_spline(numpy.array(time_plot), numpy.array(loss_plot))(x_smooth)
        plt.plot(x_smooth, y_smooth)
        plt.savefig('plot.jpg')
        plt.show()

        return time_plot, loss_plot


    # 预测结果，传入参数需要得到预测值的个数
    def center_predict(self, node_id, time=150):
        torch.save(self.model, "model.pkl")
        # self.train()
        # 返回二维数组，索引减一代表节点号对应的预测结果值
        predicts = []
        # 取出最后一个时刻的特征
        cur_test_x = self.var_x[-1:]
        # time是需要的结果个数
        for _ in range(time):
            predict = self.model(cur_test_x)
            # for j in range(len(predict[0][0])):
            #     predict[0][0][j] = cur_test_x[0][0][j: j + look_back].mean() * 0.8 + predict[0][0][j] * 0.2

            predicts.append(predict.data.numpy().reshape(-1).tolist())
            # 更改下一时刻的特征
            cur = []
            for i in range(1, Nums*look_back+1):
                if i % look_back:
                    cur.append(cur_test_x[0][0][i])
                else:
                    cur.append(predict[0][0][i//look_back-1].data)
            cur_test_x = torch.from_numpy(numpy.array(cur)).view(-1, 1, look_back*Nums)

        predicts = scaler.inverse_transform(numpy.array(predicts).reshape(Nums, -1))

        result = numpy.zeros((30, time))
        for i, node in enumerate(self.Node):
            result[node] = predicts[i]
        return result[node_id]

tp = TrainPredict()
time_plot, loss_plot = tp.train()

result = tp.center_predict(1, 150)
r = numpy.concatenate((d1, result), axis=0)


