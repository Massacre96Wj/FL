# -*- coding: UTF-8 -*-
"""
@Author ：WangJie
@Date   ：2020/11/29  10:10
@Desc   ：
"""

import torch
from torch import nn
import numpy
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from functools import reduce
from log import logger

look_back = 4
EPOCH = 10
SIZE = 100
dataset = None

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

numpy.random.seed(7)
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
dataset[1] = numpy.random.poisson(lam=20, size=SIZE)
dataset[2] = numpy.random.poisson(lam=10, size=SIZE)
dataset[4] = numpy.random.poisson(lam=30, size=SIZE)

# 判断参与的节点有哪些
def join_train(dataset):
    cur_dataset = []
    cur_node = []
    for i, data in enumerate(dataset):
        if not (data == 0).all():
            cur_dataset.append(data.tolist())
            cur_node.append(i)
    return numpy.array(cur_dataset), cur_node, len(cur_node)
dataset, Node, Nums = join_train(dataset)

# 创建数据
def load_data1(dataset):
    # dataset = numpy.random.poisson(lam=lambd, size=300).reshape(-1, 1)
    # 将整型变为float
    dataset = dataset.astype('float32')
    # normalize the dataset
    dataset = scaler.fit_transform(dataset)
    data_X, data_Y = create_dataset(dataset, look_back)
    return data_X, data_Y

# path = r'./possion.csv'
# path1 = r'./possion1.csv'
# path2 = r'./possion2.csv'
# path3 = r'./possion3.csv'

# 一、数据处理
dfs_x, dfs_y = [], []
for data in dataset:
    data_X, data_Y = load_data1(data.reshape(-1, 1))
    dfs_x.append(data_X)
    dfs_y.append(data_Y.reshape(-1, 1))

data_X = reduce(lambda left, right: numpy.concatenate((left, right), axis=1), dfs_x)
data_Y = reduce(lambda left, right: numpy.concatenate((left, right), axis=1), dfs_y)
train_X = data_X
train_Y = data_Y
train_X = numpy.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
train_Y = train_Y.reshape(-1, 1, train_Y.shape[1])
var_x = torch.from_numpy(train_X)
var_y = torch.from_numpy(train_Y)

'''
train_size = SIZE - look_back
test_size = len(dataset) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]

test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

train_X = numpy.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = numpy.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
train_Y = train_Y.reshape(-1, 1, train_Y.shape[1])

var_x = torch.from_numpy(train_X)
var_y = torch.from_numpy(train_Y)
'''

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

model = LSTM(train_X.shape[-1], 64, Nums, 1)
print(model)
loss_fun = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(1, EPOCH+1):
    logger.info("Epoch: %d......" % epoch)
    for t in range(50):
        # 前向传播
        out = model(var_x)
        loss = loss_fun(out, var_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

'''
    var_testX = torch.from_numpy(test_X)
    pred_testY = model(var_testX).squeeze(1).detach().numpy()
    logger.info("本轮测试误差：{:.3f}".format(numpy.nanmean(evluation(pred_testY.reshape(-1), test_Y.reshape(-1)))))

    for i in range(1):
        plt.figure()
        plt.plot(pred_testY[i], 'r')
        plt.plot(test_Y[i], 'b')
        plt.title("result of traffic")
        plt.show()
'''

# 预测结果，传入参数需要得到预测值的个数
def center_predict(time):
    # 返回二维数组，索引减一代表节点号对应的预测结果值
    predicts = []
    # 取出最后一个时刻的特征
    cur_test_x = var_x[-1:]
    # time是需要的结果个数
    for _ in range(time):
        predict = model(cur_test_x)
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
    print(predicts.shape)
    result = numpy.zeros((30, 100))
    for i, node in enumerate(Node):
        result[node] = predicts[i]
    return result
result = center_predict(100)
