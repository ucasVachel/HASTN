import torch
import torch.nn as nn

from models import HAST_item


class HASTblock(nn.Module):

    def __init__(self, args, blocks, n_vertex, biases_list):
        super(HASTblock, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(HAST_item.hast(args.Kt, n_vertex, blocks[l][-1], blocks[l+1], args.act_func, args.droprate,
                                               inputs_dim=64, nb_classes=12,
                                               nb_nodes=207, attn_drop=0.0,
                                               ffd_drop=0.0, bias_mat_list=biases_list, hid_units=[8],
                                               n_heads=[4, 1],
                                               activation=nn.ELU(), residual=False, batch_size=32
                                               ))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = HAST_item.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU()
            self.silu = nn.SiLU()
            self.do = nn.Dropout(p=args.droprate)

    def forward(self, input):
        # 增加一个维度
        x = input.unsqueeze(1)
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        
        return x.squeeze(2).permute(0, 2, 1)
