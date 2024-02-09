import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x = x

        return x


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1,
                 bias=True):
        if enable_padding == True:
            self.__padding = (kernel_size - 1) * dilation
        else:
            self.__padding = 0
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=self.__padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]

        return result


class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1,
                 bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0,
                                           dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)

        return result


class TemporalConvLayer(nn.Module):

    def __init__(self, Kt, c_in, c_out, n_vertex, act_func):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.n_vertex = n_vertex
        self.filter_convs = CausalConv2d(in_channels=c_in, out_channels=c_out, kernel_size=(Kt, 1),
                                         enable_padding=False, dilation=1)
        self.gate_convs = CausalConv2d(in_channels=c_in, out_channels=c_out, kernel_size=(Kt, 1),
                                       enable_padding=False, dilation=1)
        self.act_func = act_func
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        # x_in = self.align(x)[:, :, self.Kt - 1:, :]
        filter = self.filter_convs(x)
        filter = torch.tanh(filter)
        gate = self.gate_convs(x)
        gate = torch.sigmoid(gate)
        x = filter * gate
        # x = x + x_in
        x = self.bn(x)
        x = self.relu(x)

        return x


class Attn_head(nn.Module):
    def __init__(self, in_channel, out_sz, bias_mat, in_drop=0.0, coef_drop=0.0, activation=None,
                 residual=False, return_coef=False):
        super(Attn_head, self).__init__()
        self.bias_mat = bias_mat
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.return_coef = return_coef
        self.conv1 = nn.Conv2d(in_channel, out_sz, 1, bias=False)
        self.conv2_1 = nn.Conv2d(out_sz, 1, 1, bias=False)
        self.conv2_2 = nn.Conv2d(out_sz, 1, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.in_dropout = nn.Dropout(in_drop)
        self.coef_dropout = nn.Dropout(coef_drop)
        self.activation = activation

    def forward(self, x):

        seq = x.float()
        if self.in_drop != 0.0:
            seq = self.in_dropout(x)
            seq = seq.float()
        seq_fts = self.conv1(seq)
        # print("seq_fts.shape:",seq_fts.shape)
        f_1 = self.conv2_1(seq_fts)
        # print(f_1.shape)
        f_2 = self.conv2_1(seq_fts)
        # print(f_2.shape)
        logits = f_1 + torch.transpose(f_2, 3, 1)
        logits = self.leakyrelu(logits)
        # print("logis.shape:",logits.shape)
        logits = logits.permute(0, 2, 1, 3)
        coefs = self.softmax(logits + self.bias_mat.unsqueeze(0).float())
        # print("coefs.shape:",coefs.shape)
        if self.coef_drop != 0.0:
            coefs = self.coef_dropout(coefs)
        if self.in_drop != 0.0:
            seq_fts = self.in_dropout(seq_fts)
        ret = torch.matmul(coefs, seq_fts.permute(0, 2, 3, 1))
        ret = ret.permute(0, 3, 1, 2)
        # print("ret.shape:",ret.shape)
        if self.return_coef:
            return self.activation(ret), coefs
        else:
            return self.activation(ret)  # activation


class hast(nn.Module):

    def __init__(self, Kt, n_vertex, last_block_channel, channels, act_func, droprate,
                 inputs_dim, nb_classes, nb_nodes, attn_drop, ffd_drop,
                 bias_mat_list, hid_units, n_heads, activation, residual, batch_size):
        super(hast, self).__init__()
        self.Kt = Kt
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        # self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)
        self.cjh = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1))
        # self.cjh2 = nn.Linear(64, 16)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

        # han
        self.inputs_dim = inputs_dim
        self.nb_classes = nb_classes
        self.nb_nodes = nb_nodes
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.bias_mat_list = bias_mat_list
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.activation = activation
        self.residual = residual
        self.mp_att_size = 128
        self.layers = self._make_attn_head()
        self.batch_size = batch_size

    def _make_attn_head(self):
        layers = []
        for biases in self.bias_mat_list:
            layers.append(
                Attn_head(in_channel=self.inputs_dim, out_sz=self.hid_units[0], bias_mat=biases, in_drop=self.ffd_drop,
                          coef_drop=self.attn_drop, activation=self.activation, residual=self.residual))
        print("当前有{}个注意力头".format(len(layers)))
        return nn.Sequential(*list(m for m in layers))

    def forward(self, x):  # (32, 1, 12, 207)      # (32, 64, 8, 207)
        x = self.tmp_conv1(x)  # (32, 64, 10, 207)     # (32, 64, 6, 207)

        embed_list = []
        for i, (inputs, biases) in enumerate(zip([x, x], self.bias_mat_list)):

            attns = []
            for _ in range(self.n_heads[0]):
                attns.append(self.layers[i](inputs))
            h_1 = torch.cat(attns, dim=1)
            embed_list.append(h_1)

        H_embed = torch.cat(embed_list, dim=1)
        H_embed = H_embed + x
        H_embed = self.relu(H_embed)
        x = self.cjh(H_embed)  # .permute(0, 3, 1, 2)

        x = self.relu(x)
        x = self.tmp_conv2(x)  # (32, 64, 8, 207)      (32, 64, 4, 207)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)

        return x


class OutputBlock(nn.Module):

    def __init__(self, Ko, last_block_channel, channels, end_channel, n_vertex, act_func, bias, droprate):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Ko, last_block_channel, channels[0], n_vertex, act_func)
        self.fc1 = nn.Linear(in_features=channels[0], out_features=channels[1], bias=bias)
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel, bias=bias)
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]])
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):  # (32, 64, 4, 207)
        x = self.tmp_conv1(x)  # (32, 128, 1, 207)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))  # (32, 1, 207, 128)
        x = self.fc1(x)  # (32, 1, 207, 128)
        x = self.relu(x)
        x = self.fc2(x).permute(0, 3, 1, 2)  # (32, 12, 1, 207)

        return x
