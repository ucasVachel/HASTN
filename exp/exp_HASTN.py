import os, json
import torch
import numpy as np
from models.HAST import HASTblock
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchmetrics
from prettytable import PrettyTable
from datetime import datetime

import argparse
import time
from data.gcn_utils import load_pickle,load_dataset, get_dist_matrix, get_undirect_adjacency_matrix, load_adj
from data.generate_dated_data_new import generate_train_val_test
from utils.metrics import MAE, MSE, RMSE, MAPE
from utils.tools import EarlyStopping, adjust_learning_rate
from utils import utility

def print_table(val_loss_mae, val_loss_rmse, val_loss_mape, test_loss_mae, test_loss_rmse, test_loss_mape):
    # 创建一个表格对象
    table = PrettyTable()

    # 设置表格的列标题
    table.field_names = ['Metric', 'Val-MAE', 'Val-RMSE', 'Val-MAPE', '  ', 'Test-MAE', 'Test-RMSE', 'Test-MAPE']

    # 添加数据
    table.add_row(['Value', round(val_loss_mae, 4), round(val_loss_rmse, 4), round(val_loss_mape, 4), '  ', round(test_loss_mae, 4), round(test_loss_rmse, 4), round(test_loss_mape, 4)])

    # 打印表格
    print(table)


def adj_to_bias(adj, sizes, nhood=1):
    # adj -> [1,3025,3025]
    # sizes -> [3025]
    nb_graphs = adj.shape[0]  # 1
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        # 对角元素上变为1
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    # 未连接的节点值设置为一个很小的负数，用于在注意力计算时屏蔽掉
    return -1e9 * (1.0 - mt)


def get_parameters():
    parser = argparse.ArgumentParser(description='HASTN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='metr-la', choices=['metr-la', 'pems-bay', 'pemsd7-m'])
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=12,
                        help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv',
                        choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap',
                        choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10000, help='epochs, default as 10000')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([12])

    return args, device, blocks

class Exp_HASTN(object):
    def __init__(self, config):
        self.config = config
        self.data_config = config['Data']
        self.model_config = config['Model']
        self.training_config = config['Training']

        # data config.
        self.root_path = self.data_config['root_path']
        self.data_path = self.data_config['data_path']
        self.dataset_name = self.data_config['dataset_name']
        self.data_split = json.loads(self.data_config['data_split'])  # load list
        self.dist_path = self.data_config['dist_path']
        self.adjdata = self.data_config['adjdata']
        self.adjtype = self.data_config['adjtype']

        # model config
        self.model_name = self.model_config['model_name']
        self.n_his = int(self.model_config['n_his'])
        self.n_pred = int(self.model_config['n_pred'])
        self.Kt = int(self.model_config['Kt'])
        self.stblock_num = int(self.model_config['stblock_num'])
        self.act_func = self.model_config['time_intvl']
        self.Ks = int(self.model_config['Ks'])
        self.enable_bias = json.loads(self.model_config['enable_bias'].lower())

        # training config
        self.use_gpu = json.loads(self.training_config['use_gpu'].lower())
        self.gpu = int(self.training_config['gpu'])
        self.save_path = self.training_config['save_path']
        self.learning_rate = float(self.training_config['learning_rate'])
        self.lr_type = self.training_config['lr_type']
        self.patience = int(self.training_config['patience'])
        self.use_amp = json.loads(self.training_config['use_amp'].lower())
        self.batch_size = int(self.training_config['batch_size'])
        self.train_epochs = int(self.training_config['train_epochs'])
        # self.step_size = int(self.training_config['step_size'])
        # self.gamma = int(self.training_config['gamma'])

        self.device = self._acquire_device()

        # result save
        testing_info = "model_{}_{}".format(
            self.model_name,
            self.dataset_name
        )
        self.save_path = self.save_path + testing_info + '/'

        self.model = self._build_model().to(self.device)

        # loading train to #
        # best_model_path = self.save_path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        # full_dataset: (N, D)
        stat_file = os.path.join(self.root_path, "Time_sequence.npz")

        # 划分数据集
        # generate_train_val_test(stat_file, train_val_test_split=self.data_split)

        # read the pre-processed & splitted dataset from file
        self.dataloader = load_dataset(stat_file, self.batch_size)
        self.train_loader = self.dataloader['train_loader']
        self.vali_loader = self.dataloader['val_loader']
        self.test_loader = self.dataloader['test_loader']
        self.max_speed = self.dataloader['max_speed']

    def _build_model(self):
        _, _, _, adj_mx = load_adj(self.adjdata, self.adjtype)
        # 加载参数
        args, device, blocks = get_parameters()
        # gso = utility.calc_gso(adj_mx, args.gso_type)
        # if args.graph_conv_type == 'cheb_graph_conv':
        #     gso = utility.calc_chebynet_gso(gso)
        # gso = gso.toarray()
        # gso = gso.astype(dtype=np.float32)
        # args.gso = torch.from_numpy(gso).to(device)

        self.gis_adj = load_pickle(self.data_config['gis_adj'])
        adj_list = [adj[np.newaxis] for adj in [adj_mx, self.gis_adj]]

        self.biases_list = [
            torch.transpose(torch.from_numpy(adj_to_bias(adj, [adj_mx.shape[0]], nhood=1)), 2, 1).to(self.device) for
            adj in
            adj_list]

        model = HASTblock(args, blocks, adj_mx.shape[0], self.biases_list)
        return model


    def _acquire_device(self):
        if self.use_gpu:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
            device = torch.device('cuda:{}'.format(self.gpu))
            print('Use GPU: cuda:{}'.format(self.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return model_optim


    def vali(self, vali_loader):
        self.model.eval()
        totle_moss_mse = []
        total_loss_mae = []
        total_loss_rmse = []
        total_loss_mape = []

        for i, (batch_x, batch_dateTime, batch_y) in enumerate(vali_loader.get_iterator()):
            batch_x = torch.Tensor(batch_x).to(self.device)  # (B, L, D)
            batch_y = torch.Tensor(batch_y).to(self.device)  # (B, L, W)
            batch_y = torch.transpose(batch_y, 1, 2) # (B, W, L)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x)  # (B, L, D) -> (B, L, W)
            else:
                outputs = self.model(batch_x)

            outputs = torch.mul(outputs, torch.Tensor(self.max_speed)).cpu().detach().numpy()  # (B, L, D)
            batch_y = torch.mul(batch_y, torch.Tensor(self.max_speed)).cpu().detach().numpy()

            loss_mse = MSE(outputs, batch_y)
            loss_mae = MAE(outputs, batch_y)
            loss_rmse = RMSE(outputs, batch_y)
            loss_mape = MAPE(outputs, batch_y)

            totle_moss_mse.append(loss_mse)
            total_loss_mae.append(loss_mae)
            total_loss_rmse.append(loss_rmse)
            total_loss_mape.append(loss_mape)

        total_loss_mse = np.average(totle_moss_mse)
        total_loss_mae = np.average(total_loss_mae)
        total_loss_rmse = np.average(total_loss_rmse)
        total_loss_mape = np.average(total_loss_mape)
        self.model.train()
        # return total_loss
        return total_loss_mse, total_loss_rmse, total_loss_mae, total_loss_mape

    def train(self):


        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        time_now = time.time()

        train_steps = self.train_loader.size
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        model_optim = self._select_optimizer()

        if self.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # 打乱数据顺序
            self.train_loader.shuffle()
            for i, (batch_x, batch_dateTime, batch_y) in enumerate(self.train_loader.get_iterator()):

                iter_count += 1

                model_optim.zero_grad()
                batch_x = torch.Tensor(batch_x).to(self.device)  # (B, L, D)
                batch_y = torch.Tensor(batch_y).to(self.device)  # (B, L, M)
                batch_y = torch.transpose(batch_y, 1, 2)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)  # (B, L, M)
                else:
                    outputs = self.model(batch_x)

                outputs = torch.mul(outputs, torch.Tensor(self.max_speed).to(self.device))  # (B, L, D)

                batch_y = torch.mul(batch_y, torch.Tensor(self.max_speed).to(self.device))

                loss = F.mse_loss(outputs, batch_y)  # [N, L, M]
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.train_epochs - epoch) * (train_steps // self.batch_size) - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            val_loss_mse, val_loss_rmse, val_loss_mae, val_loss_mape = self.vali(self.vali_loader)
            test_loss_mse, test_loss_rmse, test_loss_mae, test_loss_mape = self.vali(self.test_loader)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss mae: {2:.7f} ".format(
                    epoch + 1, train_steps, train_loss))
            print_table(val_loss_mae, val_loss_rmse, val_loss_mape, test_loss_mae, test_loss_rmse, test_loss_mape)

            early_stopping(val_loss_mse, self.model, self.save_path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.learning_rate, self.lr_type)

        best_model_path = self.save_path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return

    def test(self):
        test_loader = self.dataloader['test_loader']
        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_dateTime, batch_y) in enumerate(test_loader.get_iterator()):
            batch_x = torch.Tensor(batch_x).to(self.device)  # (B, L, D)
            batch_y = torch.Tensor(batch_y).to(self.device)  # (B, L, M)
            batch_y = torch.transpose(batch_y, 1, 2)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x)  # (B, L, D), -> [N, L, W]
            else:
                outputs = self.model(batch_x)

            outputs = torch.mul(outputs, torch.Tensor(self.max_speed).to(self.device))  # (B, L, D)

            batch_y = torch.mul(batch_y, torch.Tensor(self.max_speed).to(self.device))

            preds.append(outputs.cpu().detach().numpy())
            trues.append(batch_y.cpu().detach().numpy())

        # print('test shape 1:', preds.shape, trues.shape)
        outputs = np.concatenate(preds, axis=0)  # [B, L, D] -> [N, L, D]
        batch_y = np.concatenate(trues, axis=0)  # [B, L, D] -> [N, L, D]
        print('test shape:', outputs.shape, batch_y.shape)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        loss_mae = MAE(outputs, batch_y)
        loss_rmse = RMSE(outputs, batch_y)
        loss_mape = MAPE(outputs, batch_y)

        print('[Average value] mae:{}, rmse:{}, mape:{}'.format(loss_mae, loss_rmse, loss_mape))

        np.save(self.save_path + 'metrics.npy', np.array([loss_mae, loss_rmse, loss_mape]))
        np.save(self.save_path + 'pred.npy', preds)
        np.save(self.save_path + 'true.npy', trues)

        return

    def predict(self, point, times):
        test_loader = self.dataloader['test_loader']
        self.model.eval()

        preds = []
        trues = []

        # point = 0
        t = []
        t_speed = []
        y_speed = []
        ylabel_flag = True
        for i, (batch_x, batch_dateTime, batch_y) in enumerate(test_loader.get_iterator()):

            if i < times:
                continue
            if i == times+8:
                break
            batch_x = torch.Tensor(batch_x).to(self.device)  # (B, L, D)
            batch_y = torch.Tensor(batch_y).to(self.device)  # (B, L, M)
            batch_y = torch.transpose(batch_y, 1, 2)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x)  # (B, L, D), -> [N, L, W]
            else:
                outputs = self.model(batch_x)

            outputs = torch.mul(outputs, torch.Tensor(self.max_speed).to(self.device))  # (B, L, D)

            batch_y = torch.mul(batch_y, torch.Tensor(self.max_speed).to(self.device))

            preds.append(outputs.cpu().detach().numpy())
            trues.append(batch_y.cpu().detach().numpy())

            for bs in range(0, 32):
                if bs == 0 and ylabel_flag:
                    t.append(str(batch_dateTime[bs, 0])[0:10]+str(batch_dateTime[bs, 0])[11:16])
                    ylabel_flag = False
                else:
                    t.append(str(batch_dateTime[bs, 0])[11:16])
                t_speed.append(batch_y[bs, point, 0].item())
                y_speed.append(outputs[bs, point, 0].item())


        # print('test shape 1:', preds.shape, trues.shape)
        outputs = np.concatenate(preds, axis=0)  # [B, L, D] -> [N, L, D]
        batch_y = np.concatenate(trues, axis=0)  # [B, L, D] -> [N, L, D]
        print('test shape:', outputs.shape, batch_y.shape)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        loss_mae = MAE(outputs, batch_y)
        loss_rmse = RMSE(outputs, batch_y)
        loss_mape = MAPE(outputs, batch_y)

        print('[Average value] mae:{}, rmse:{}, mape:{}'.format(loss_mae, loss_rmse, loss_mape))

        np.save(self.save_path + 'metrics.npy', np.array([loss_mae, loss_rmse, loss_mape]))
        np.save(self.save_path + 'pred.npy', preds)
        np.save(self.save_path + 'true.npy', trues)

        return t, t_speed, y_speed