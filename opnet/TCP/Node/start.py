# -*- coding: UTF-8 -*-
"""
@Author ：WangJie
@Date   ：2020/10/23  14:38
@Desc   ：
"""
import os
import threading
from socket import *
import time

BUFSIZ = 1024
cnt = 0
def send(HOST, PORT):
    ADDR = (HOST, PORT)
    tcpCliSock = socket(AF_INET, SOCK_STREAM)
    tcpCliSock.connect(ADDR)
    data = input()
    tcpCliSock.send(data.encode())
    data = tcpCliSock.recv(BUFSIZ)
    print(data.decode('utf-8'))
    tcpCliSock.close()

def startCenter():
    os.system('python E:\pythonDeme\FL过程\opnet\TCP\center.py')

if __name__ == '__main__':
    while True:
        # t = threading.Thread(target=startCenter())
        t1 = threading.Thread(target=send, args=('127.0.0.1', 8881))
        t2 = threading.Thread(target=send, args=('127.0.0.1', 8882))
        t1.setDaemon(True)
        t2.setDaemon(True)
        # t.start()
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        print('end')
        time.sleep(1200)
