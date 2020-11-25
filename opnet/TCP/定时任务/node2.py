# -*- coding: UTF-8 -*-
"""
@Author ：WangJie
@Date   ：2020/10/23  22:12
@Desc   ：
"""
import os
import threading
from socket import *
import time

BUFSIZ = 1024
cnt = 0
def send(HOST, PORT,data):
    ADDR = (HOST, PORT)
    tcpCliSock = socket(AF_INET, SOCK_STREAM)
    tcpCliSock.connect(ADDR)
    data = '2Y'
    from FL过程.opnet.TCP.Node2 import error
    if error < 0.1:
        data[1] = 'N'
    tcpCliSock.send(data.encode())
    data = tcpCliSock.recv(BUFSIZ).decode('utf-8')
    tcpCliSock.close()
    if 'OK' in data:
        os.system(r'python E:\pythonDeme\FL过程\opnet\TCP\Node2.py')
    global timer
    timer = threading.Timer(1800, send, args=('127.0.0.1', 8088, data))
    timer.start()

if __name__ == '__main__':
    timer = threading.Timer(2, send, args=('127.0.0.1', 8088, '2Y'))
    timer.start()