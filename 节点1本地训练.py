# -*- coding: UTF-8 -*-
"""
@Author ：WangJie
@Date   ：2020/6/2  19:11
@Desc   ：
"""
import pickle
import socket
import time

import torch
import torch.nn.functional as F  #nn神经网络模块
from torch.autograd import Variable
import matplotlib.pyplot as plt

EPOCH = 500

def log(info):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))

# 构造数据特征
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)      #unsqueeze一维数据变为二维数据，shape(100, 1)

# 构造数据标签
y = x.pow(2) + 0.2*torch.rand(x.size())   #rand返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数,randn返回高斯分布，服从均值0，方差1

# def save():
# 第一个子模型
net = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),        #大写，代表是一个类
    torch.nn.Linear(10, 1)
    )
# print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_fun = torch.nn.MSELoss()
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

for epoch in range(1, EPOCH+1):
    log("本地第"+str(epoch)+"轮训练中...")
    time.sleep(5)
    for t in range(100):
        prediction = net(x)
        loss = loss_fun(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 上传开始
    # print(net.state_dict())
    log("建立连接并上传......")
    # 定义中心节点的地址端口号
    host = '127.0.0.1'
    port = 9999
    # 序列化
    data = pickle.dumps(net.state_dict())
    s.sendto(data, (host, port))
    # 等待待收
    log("等待接收......")
    try:
        s.settimeout(30)
        global_state_dict = pickle.loads(s.recv(1024 * 100))
    except:
        s.sendto(data, (host, port))
    # print(global_state_dict)
    # 重新加载全局参数
    net.load_state_dict(global_state_dict)
log("训练完毕，关闭连接")
s.close()
log("开始本地预测......")
with torch.no_grad():
    pre = net(x)