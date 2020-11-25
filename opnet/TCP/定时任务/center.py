# -*- coding: UTF-8 -*-
"""
@Author ：WangJie
@Date   ：2020/10/23  22:11
@Desc   ：
"""
import os
from socket import *
import time, threading
from time import ctime

def startCenter():
    os.system('python E:\pythonDeme\FL过程\opnet\TCP\Center.py')
    global timer
    # 重复构造定时器
    timer = threading.Timer(1800, startCenter)
    timer.start()

def response(sock, addr):
    data = sock.recv(1024).decode()
    if data[1] == 'Y':
        data = data[0] + ' OK'
    sock.send(('[%s]  %s' % (ctime(), data)).encode())
    sock.close()

timer = threading.Timer(2, startCenter)
timer.start()

HOST = '127.0.0.1'
PORT = 8088
BUFSIZ = 1024
ADDR = (HOST, PORT)

tcpSerSock = socket(AF_INET,SOCK_STREAM)
tcpSerSock.bind(ADDR)
tcpSerSock.listen(5)

while True:
    tcpCliSock, addr = tcpSerSock.accept()
    print('connnecting from:', addr)
    t = threading.Thread(target=response, args=(tcpCliSock, addr))
    t.start()

tcpSerSock.close()
