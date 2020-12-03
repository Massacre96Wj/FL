# -*- coding: UTF-8 -*-
"""
@Author ：WangJie
@Date   ：2020/11/26  22:51
@Desc   ：
"""
# 一、数据产生
import random
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

# 泊松
SAMPLE_SIZE = 1000
lambd = int(input("输入possion分布lambda: "))
res_1 = np.random.poisson(lam=lambd, size=SAMPLE_SIZE)  # lam为λ size为k

# 均匀分布
a = lambd//2
b = SAMPLE_SIZE
res_2 = [random.uniform(a, 2*lambd) for _ in range(1, SAMPLE_SIZE)]+[0]

# 求和混合流
res = res_2+res_1
print(res[:10])

def evluation(real, pred):
    return np.abs(real-pred)/real
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
scaler = MinMaxScaler(feature_range=(0, 1))

# 处理数据
ts = pd.Series(np.log(res+1).reshape(-1), index=range(len(res)))
ts.fillna(np.nanmean(np.array(ts)))
#平稳检验
def testStationarity(timeseries):
    #plot rolling statistics:
    plt.figure()
    plt.xticks(rotation=60)
    plt.plot(timeseries, color = 'blue',label='Original')

    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries,autolag = 'AIC')
    #dftest的输出前一项依次为检测值，p值，滞后数，使用的观测数，各个置信度下的临界值
    dfoutput = pd.Series(dftest[0:4],index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical value (%s)' %key] = value

    return dfoutput

#自相关和偏相关
def draw_acf_pacf(ts, lags=31):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=31, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=31, ax=ax2)
    plt.show()

#移动平均法
def draw_trend(timeSeries, size):
    plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = pd.ewma(timeSeries, span=size)
    #标准差
    rolstd = pd.rolling_std(timeSeries, window=24)
    plt.plot(rolstd, color='yellow', label = 'Rolling Std')
    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()
    return rol_weighted_mean

# 返回包含三个部分 trend（趋势部分）， seasonal（季节性部分） 和residual (残留部分)
def decompose(timeseries):
    decomposition = seasonal_decompose(timeseries, freq=3)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    # plt.subplot(411)
    # plt.xticks(rotation=15)
    # plt.plot(timeseries, label='Original')
    # plt.legend(loc='best')
    # plt.subplot(412)
    # plt.xticks(rotation=15)
    # plt.plot(trend, label='Trend')
    # plt.legend(loc='best')
    # plt.subplot(413)
    # plt.xticks(rotation=15)
    # plt.plot(seasonal,label='Seasonality')
    # plt.legend(loc='best')
    # plt.subplot(414)
    # plt.xticks(rotation=15)
    # plt.plot(residual, label='Residuals')
    # plt.legend(loc='best')
    # plt.tight_layout()
    return trend , seasonal, residual

# 提取特征
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# 定义训练集大小
look_back = 4
SIZE = int(len(res)*0.9)
train_size = SIZE - look_back

trend, seasonal, residual = decompose(ts)

r = int(input("请输入0，1（分别代表加随机项和不加随机项）："))
while 1:
    if r == 1:
        ts = trend + seasonal
        break
    elif r == 0:
        ts = trend + seasonal + residual
        break
    else:
        r = int(input("重新输入0，1（分别代表加随机项和不加随机项）："))

ts = ts.fillna(np.nanmean(np.array(ts)))
pd.DataFrame(ts).to_csv("./possion.csv", index=False)

select = int(input("请输入1，2, 3, 4，5 (分别代表ARIAM, LSTM, Ridge, SVM，RF):"))
while 1:
    # ARIMA          0.111
    if select == 1:
        ts1 = ts[:900]
        ts2 = ts[-100:]

        X = ts1.values
        Y = ts2.values
        train, test = X[0:int(len(X))], Y[0:int(len(Y))]
        history = [x for x in train]
        predictions = list()

        model = ARIMA(history, order=(3, 0, 3))
        model_fit = model.fit(disp=0)
        predictions = model_fit.predict(0, len(test) - 1)
        test = np.exp(test) - 1
        prediction = np.exp(np.array(predictions).reshape(-1)) - 1
        error = evluation(test, prediction)
        print('Test MRE: %.3f' % error.mean())

        plt.figure(1)
        plt.plot(prediction, 'r', label='prediction')
        plt.plot(test, 'b', label='real')
        plt.title("Result Of Prediction")
        plt.xlabel("time")
        plt.ylabel("NomalTraffic")
        plt.legend(loc='best')
        plt.show()
        break
    # LSTM           0.081
    elif select == 2:
        def fl_data_load(path):
            dataframe = pd.read_csv(path, usecols=[0], engine='python', skipfooter=3)
            dataset = dataframe.values
            # dataset = dataframe[576:1416].values
            # 将整型变为float
            dataset = dataset.astype('float32')
            np.random.seed(7)
            dataset = scaler.fit_transform(dataset)
            data_X, data_Y = create_dataset(dataset, look_back)

            # split dataset into train and test sets
            train_size = SIZE - look_back
            test_size = len(dataset) - train_size
            train_X = data_X[:train_size]
            train_Y = data_Y[:train_size]
            test_X = data_X[train_size:]
            test_Y = data_Y[train_size:]
            # 投入到 LSTM 的 X 需要有这样的结构： [samples, time steps, features]，所以做一下变换
            # reshape dataset  input to be [samples, time steps, features]
            train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
            train_Y = train_Y.reshape(-1, 1, 1)
            test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

            var_x = torch.from_numpy(train_X)
            var_y = torch.from_numpy(train_Y)
            var_testX = torch.from_numpy(test_X)

            return var_x, var_y, var_testX, test_Y

        path = r"./possion.csv"

        var_x, var_y, var_testX, test_Y = fl_data_load(path)

        class LSTM(torch.nn.Module):
            def __init__(self, input_size=2, hidden_size=4, output_size=1, num_layer=1):
                super(LSTM, self).__init__()
                self.layer1 = torch.nn.LSTM(input_size, hidden_size, num_layer)
                self.layer2 = torch.nn.Linear(hidden_size, output_size)

            def forward(self, x):
                x, _ = self.layer1(x)
                x = torch.relu(x)
                s, b, h = x.size()
                x = x.view(s * b, h)
                x = self.layer2(x)
                x = x.view(s, b, -1)
                return x

        # 二、模型构建
        model = LSTM(look_back, 4, 1, 1)
        print(model)
        loss_fun = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

        EPOCH = 5
        # 三、开始训练
        for epoch in range(1, EPOCH + 1):
            print("第 %s 轮开始训练!" % str(epoch))
            # 第一个网络
            for t in range(100):
                # 前向传播
                out = model(var_x)
                loss = loss_fun(out, var_y)
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 可视化
            model.eval()
            pred_testY = model(var_x)
            test_Y_origin = scaler.inverse_transform(var_y.reshape(-1, 1).data.numpy())
            pred_testY_origin = scaler.inverse_transform(pred_testY.view(-1, 1).data.numpy())
            plt.ion()
            plt.clf()
            plt.figure(1, figsize=(8, 5))
            plt.grid(True, linestyle='--')
            plt.plot(test_Y_origin[-168:], color='red', label='真实值')
            plt.plot(pred_testY_origin[-168:], color='blue', label='预测值')
            plt.title("Node1")
            plt.legend(loc='best')
            plt.xlabel('时间')
            plt.ylabel('业务量')
            plt.pause(2)
            plt.show()
            model.train()
        with torch.no_grad():
            model.eval()
            pred_testY = model(var_testX)
            pred_testY = pred_testY.view(-1, 1).data.numpy()
            test_Y_origin = scaler.inverse_transform(test_Y.reshape(-1, 1))
            pred_testY_origin = scaler.inverse_transform(pred_testY)
            test_Y_origin = np.exp(test_Y_origin) - 1
            pred_testY_origin = np.exp(pred_testY_origin) - 1
            error = evluation(test_Y_origin, pred_testY_origin).mean()
            plt.ioff()
            plt.figure(1, figsize=(8, 5))
            plt.clf()
            plt.grid(True, linestyle='--')
            plt.plot(test_Y_origin, color='red', label='真实值')
            plt.plot(pred_testY_origin, color='blue', label='预测值')
            # plt.title("result of traffic")
            plt.legend(loc='best')
            plt.xlabel('时间')
            plt.ylabel('业务量')
            plt.show()
            print("相对误差：{:.3f}".format(error))
        break
    # 回归           0.081
    elif select == 3:
        data_X, data_Y = create_dataset(np.array(ts).reshape(-1, 1), 4)
        x_train, x_test, y_train, y_test = data_X[:train_size], data_X[train_size:], data_Y[:train_size], data_Y[train_size:]
        clf = Ridge(alpha=1.0, fit_intercept=True)

        # w: clf.coef_, b:clf.intercept_
        clf.fit(x_train, y_train)
        clf.score(x_test, y_test)

        y_pre = clf.predict(x_test)
        plt.figure()
        plt.plot(np.exp(y_test) - 1, 'b', label = 'real')
        plt.plot(np.exp(y_pre) - 1, 'r', label = 'predict')
        plt.legend(loc='upper left')
        plt.show()
        error = evluation(np.exp(y_test) - 1, np.exp(y_pre) - 1)
        print('Test MRE: %.3f' % error.mean())
        break
    # svm
    elif select == 4:
        data_X, data_Y = create_dataset(np.array(ts).reshape(-1, 1), 4)
        x_train, x_test, y_train, y_test = data_X[:train_size], data_X[train_size:], data_Y[:train_size], data_Y[
                                                                                                          train_size:]
        clf = SVR(kernel="linear")
        clf.fit(x_train, y_train)
        clf.score(x_test, y_test)

        y_pre = clf.predict(x_test)
        plt.figure()
        plt.plot(np.exp(y_test) - 1, 'b', label='real')
        plt.plot(np.exp(y_pre) - 1, 'r', label='predict')
        plt.legend(loc='upper left')
        plt.show()
        error = evluation(np.exp(y_test) - 1, np.exp(y_pre) - 1)
        print('Test MRE: %.3f' % error.mean())
        break
    # 随机森林
    elif select == 5:
        data_X, data_Y = create_dataset(np.array(ts).reshape(-1, 1), 4)
        x_train, x_test, y_train, y_test = data_X[:train_size], data_X[train_size:], data_Y[:train_size], data_Y[
                                                                                                          train_size:]
        rf = RandomForestRegressor(n_estimators=1000)
        rf.fit(x_train, y_train)
        y_pre = rf.predict(x_test)
        plt.figure()
        plt.plot(np.exp(y_test) - 1, 'b', label='real')
        plt.plot(np.exp(y_pre) - 1, 'r', label='predict')
        plt.legend(loc='upper left')
        plt.show()
        error = evluation(np.exp(y_test) - 1, np.exp(y_pre) - 1)
        print('Test MRE: %.3f' % error.mean())
        break

    else:
        select = int(input("重新输入1，2, 3, 4, 5 (分别代表ARIAM, LSTM, Ridge, SVM， RF):"))