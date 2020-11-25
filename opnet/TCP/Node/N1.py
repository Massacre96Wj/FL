# -*- coding: UTF-8 -*-
"""
@Author ：WangJie
@Date   ：2020/10/23  14:58
@Desc   ：
"""
import os
from socket import *
import time
from time import ctime

HOST = '127.0.0.1'
PORT = 8881
BUFSIZ = 1024
ADDR = (HOST, PORT)

tcpSerSock = socket(AF_INET,SOCK_STREAM)
tcpSerSock.bind(ADDR)
tcpSerSock.listen(5)
while True:
    print('waiting for connection...')
    tcpCliSock, addr = tcpSerSock.accept()
    print('...connnecting from:', addr)

    data = tcpCliSock.recv(BUFSIZ).decode()
    print(data)
    if data == 'Y':
        time.sleep(1)
        # os.system(r'python E:\pythonDeme\FL过程\opnet\TCP\Node1.py')
        data = 'Node1启动FL'
    tcpCliSock.send(('[%s]  %s' % (ctime(), data)).encode())
    tcpCliSock.close()
tcpSerSock.close()

