# -*- coding: UTF-8 -*-
"""
@Author ：WangJie
@Date   ：2020/6/2  8:39
@Desc   ：
"""
import pickle
import socket
import time

def log(info):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))

Epoch = 5

def m1(*args):
    import copy
    result = copy.deepcopy(args[0])
    for i in range(1, len(args)):
        for j in range(len(result)):
            result[j] += args[i][j]

    for i in range(len(args)):
        for j in range(len(args[0])):
            args[i][j] = result[j] / len(args)

def m2(*args):
    import copy
    result = copy.deepcopy(args[0])
    for i in range(1, len(args)):
        # print(args[i])
        for j in range(len(result)):
            for k in range(len(result[0])):
                result[j][k] += args[i][j][k]

    for i in range(len(args)):
        for j in range(len(result)):
            for k in range(len(result[0])):
                args[i][j][k] = result[j][k] / len(args)

def socket_udp_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # SOCK_DGRAM指类型是UDP
    host = '127.0.0.1'  # 监听指定的ip,host=''即监听所有的ip
    port = 9999
    # 绑定端口
    s.bind((host, port))

    res, addrs = [], []
    cnt = 1
    while True:
        log("第%d轮开始接收并计时" % cnt)
        try:
            s.settimeout(30)
            start = time.time()
            # 接收操作
            data, addr = s.recvfrom(1024*100)
            # print('Received from %s:%s' % addr)
            # print('Received data:', data)
            addrs.append(addr)
            # print(addrs)
            res.append(pickle.loads(data))
            # print(res)
            recv_time = time.time() - start
            # print(len(res))
            if len(res) >= 2 or recv_time > 2000000:
                log("第%d轮接收完毕 接收来自%d个节点的参数" % (cnt, len(res)))
                # 处理操作
                if len(res) > 1:
                    log("开始融合处理操作......")
                    time.sleep(5)
                    # res = str(sum(res))
                    for m, n in zip(res[0].values(), res[1].values()):
                        if len(m.size()) == 1:
                            m1(m, n)
                        elif len(m.size()) == 2:
                            m2(m, n)
                    # print(res[0])
                    res = pickle.dumps(res[0])
                    # 下发操作
                    log('第%d轮融合完毕，下发......' %cnt)
                    for addr in (addrs):
                        s.sendto(res, addr)
                        # s.sendto(b'%s' % res.encode('utf-8'), addr)
                # else:
                #     res = '处理完毕，关闭连接'
                #     for addr in (addrs):
                #         s.sendto(b'%s' % res.encode('utf-8'), addr)
                #     break
                res, addrs = [], []
                cnt += 1
                if cnt > Epoch:
                    log('处理完毕，关闭连接')
                    break
        except:
            log("超时重传")

def main():
    socket_udp_server()

if __name__ == '__main__':
    main()

