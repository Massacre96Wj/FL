# -*- coding: UTF-8 -*-
"""
@Author ：WangJie
@Date   ：2020/6/2  8:39
@Desc   ：
"""
import socket
import torch
import pickle

def socket_udp_client():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    net = torch.load(r"E:\pythonDeme\causality\net.pkl")
    while True:
        # data = input()
        host = '127.0.0.1'  # 客户端本机的ip
        port = 9999
        data = pickle.dumps(net.state_dict())
        # s.sendto(data.encode('utf-8'), (host, port))
        s.sendto(data, (host, port))
        print(pickle.loads(s.recv(1024*100)))

    s.close()

def main():
    socket_udp_client()

if __name__ == '__main__':
    main()