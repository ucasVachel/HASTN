import argparse, configparser
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from exp.exp_HASTN import Exp_STGCN2
from exp.exp_STGCN3 import Exp_STGCN3
from exp.exp_STGCN4 import Exp_STGCN4

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configs/PEMS_BAY_HASTN.conf', type=str,
                    help="configuration file path")
parser.add_argument("--itr", default=1, type=int,
                    help="the iteration round for the model")

args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
print(os.path.exists(args.config))

parser2 = argparse.ArgumentParser()
parser2.add_argument("--config", default='configs/PEMS_BAY_HASTG_withoutgeo.conf', type=str,
                     help="configuration file path")
parser2.add_argument("--itr", default=1, type=int,
                     help="the iteration round for the model")

args2 = parser2.parse_args()
config2 = configparser.ConfigParser()
print('Read configuration file: %s' % (args2.config))
config2.read(args2.config)
print(os.path.exists(args2.config))

parser3 = argparse.ArgumentParser()
parser3.add_argument("--config", default='configs/PEMS_BAY_HASTG_withoutroad.conf', type=str,
                     help="configuration file path")
parser3.add_argument("--itr", default=1, type=int,
                     help="the iteration round for the model")

args3 = parser3.parse_args()
config3 = configparser.ConfigParser()
print('Read configuration file: %s' % (args3.config))
config3.read(args3.config)
print(os.path.exists(args3.config))

for ii in range(args.itr):
    print('Main Interation Round {}'.format(ii))
    # torch.cuda.device_count()
    # print("xxxxx",torch.cuda.device_count())
    exp = Exp_STGCN2(config)  # set experiments
    exp2 = Exp_STGCN3(config2)  # set experiments
    exp3 = Exp_STGCN4(config3)  # set experiments
    # print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(config))
    # exp.train()

    # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(config))
    # exp.test()

    point_index = 73
    times = 50
    index = 1
    period = 20
    plt.figure(figsize=(10, 6))
    t, t_speed, y_speed = exp.predict(point_index, times)
    _, _, y_speed2 = exp2.predict(point_index, times)
    _, _, y_speed3 = exp3.predict(point_index, times)

    # 创建折线图
    print(t[0])

    plt.plot(t[index:index+period], t_speed[index:index+period], label='Ground Truth', color='r', linewidth=3)  # 绘制 t_speed 的折线
    plt.plot(t[index:index+period], y_speed[index:index+period], label='HASTN', color='b', linewidth=3)  # 绘制 y_speed 的折线
    plt.plot(t[index:index+period], y_speed2[index:index+period], label='SRS model', color='g', linestyle='--', linewidth=3)  # 绘制 y_speed 的折线
    plt.plot(t[index:index+period], y_speed3[index:index+period], label='SGS model', color='orange', linestyle=':', linewidth=3)
    plt.xticks(np.arange(0, 21, 5))
    plt.ylabel('Speed(km/h)')

    # plt.plot(t, t_speed, label='Ground Truth', color='r', linewidth=1)  # 绘制 t_speed 的折线
    # plt.plot(t, y_speed, label='HASTN', color='b', linewidth=1)  # 绘制 y_speed 的折线
    # plt.plot(t, y_speed2, label='HASTN_road', color='g', linestyle='--', linewidth=1)  # 绘制 y_speed 的折线
    # plt.plot(t, y_speed3, label='HASTN_geo', color='orange', linestyle=':', linewidth=1)
    # plt.xticks(np.arange(0, 256, 48))
    # plt.ylabel('Speed(km/h)')
    # 添加图例
    plt.legend()
    plt.title("PEMS-BAY Dataset point index:" + str(point_index))

    # 显示图形
    plt.savefig("PEMS-BAY Dataset point index:" + str(point_index) + '.png', dpi=300)
    plt.show()

    torch.cuda.empty_cache()
