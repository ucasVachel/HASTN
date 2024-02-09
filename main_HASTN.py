import argparse, configparser
import os
import torch

from exp.exp_HASTN import Exp_HASTN

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

for ii in range(args.itr):
    print('Main Interation Round {}'.format(ii))
    exp = Exp_HASTN(config) # set experiments

    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(config))
    exp.train()

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(config))
    exp.test()

    torch.cuda.empty_cache()
