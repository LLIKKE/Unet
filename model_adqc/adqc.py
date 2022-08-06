import torch as tc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
import sys
import torchvision.transforms as transforms


class adqc_conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernerl_size=4, layers=3, weight_name=None ):
        super(adqc_conv2d, self).__init__()
        self.layers = layers
        self.inchannel = in_channels
        self.outchannel = out_channels
        self.kernel = kernerl_size
        self.padding = int((self.kernel-2)/2)
        '''if self.kernel & (self.kernel - 1) != 0:  
            para_s = tc.load(weight_name)
            sheng_yv, self.weight = self.jieduan_pre_process_for_ADQC(para_s)
            '''
        weight_size = self.inchannel * (kernerl_size ** 2) * self.outchannel
        self.index = list()
        self.dims = len(bin(weight_size)) - 3
        self.index.append(self.index_gates_brick_wall_1D(self.dims, self.layers))
        self.num_gates = len(self.index[0])
        self.gates = self.initial_identity_latent_gates(
            self.num_gates)

    def jieduan_pre_process_for_ADQC(self, weight):
        # ToDo:
        dims = len(bin(weight.numel())) - 3  # weight事输入权重，.numel()计算张量里面的数的个数。bin()转化为二进制,计算2的指数
        n = 2 ** dims
        weight0 = weight.reshape(-1, )[:n].reshape([2] * dims)  # .reshape(-1:)是将全部展开成
        sheng_yu_weight = weight.reshape(-1, )[n:]  # 保留多出去的权重
        return sheng_yu_weight, weight0

    def index_gates_brick_wall_1D(self, num_qubits, num_layers, starting_gate=0):
        # todo:生产门的排序
        index = list()
        n_gate = starting_gate
        for layer in range(num_layers * 2):
            n_qubit = layer % 2
            while n_qubit + 1 < num_qubits:
                index.append([n_gate, n_qubit, n_qubit + 1])
                n_qubit += 2
                n_gate += 1
        return index

    def ADQC_evolve(self, mps, gates, gates_pos, length):
        mps1 = copy.deepcopy(mps)
        for n in range(len(gates_pos)):  # len()一共几个门
            which_gate = gates_pos[n][0]  # 第几个门，每个小列表的第一个数
            where = gates_pos[n][1:]  # 门的未知，每个小列表的后二个数
            mps1 = self.evolve_tensor_one_gate(mps1, gates[which_gate], where)
            if ((n + 1) / (length - 1)).is_integer() and (n + 1) // (length - 1) < len(gates_pos) / (length - 1):
                nn.ReLU(inplace=True)(mps1)
        return mps1

    def evolve_tensor_one_gate(self, tensor, gate, pos):
        # todo：进行门的运算
        ndim = tensor.ndimension()
        tensor = tc.tensordot(tensor, gate, [pos, [0, 1]])  # tensordot矩阵相乘，前面输入的tensor与门gate相乘，
        # [pos, [0, 1]]表示tensor的第pos的两维合并与gate的第0维与第1维合并的tensor相乘
        order = list(range(pos[0])) + [ndim - 2] + list(range(pos[0], pos[1] - 1)) + [
            ndim - 1] + list(range(pos[1] - 1, ndim - 2))
        return tensor.permute(order)  # premute对tensor多个维数进行排序比如order=（4，2，0，3，1）就是对所有维数安装列表进行重排

    def initial_identity_latent_gates(self, num):
        # todo 初始化gate参数
        gates = list()
        for n in range(num):
            gate = (tc.eye(2 * 2, dtype=tc.float32) + tc.randn(
                (2 * 2, 2 * 2), dtype=tc.float32) * 1e-12).reshape(2, 2, 2, 2)  # tc.eye创建一个d^2行d^2列的对角矩阵
            gate = nn.Parameter(gate, requires_grad=True)
            gates.append(gate)
        gates = nn.ParameterList(gates)
        return gates  # 这是多个门的合集，放到一个列表里

    def initialize_up_product_state_tensor(self, length):
        # todo： 制备初态张量
        dims = [2] * length
        mps = tc.zeros(dims, dtype=tc.float32)  # 生成一个[2,2,2,2,2....]跟目标态权重张量一样尺寸的张量
        mps[(0,) * length] = 1  # 让这个张量的第一个元素等于1
        return mps

    def ADQC_state_gates(self, index_groups):
        mps = self.initialize_up_product_state_tensor(self.dims)
        index = list()
        for n in index_groups:
            index = index + n
        mps1 = self.ADQC_evolve(mps, self.gates, index, self.dims)
        return mps1.reshape((self.outchannel, self.inchannel, self.kernel, self.kernel))

    def forward(self, x):
        n, c, h, w = x.size()
        assert c % 4 == 0
        x1 = x[:, :c // 4, :, :]
        x2 = x[:, c // 4:c // 2, :, :]
        x3 = x[:, c // 2:c // 4 * 3, :, :]
        x4 = x[:, c // 4 * 3:c, :, :]
        x1 = nn.functional.pad(x1, (1, 0, 1, 0), mode="constant", value=0)  # left top
        x2 = nn.functional.pad(x2, (0, 1, 1, 0), mode="constant", value=0)  # right top
        x3 = nn.functional.pad(x3, (1, 0, 0, 1), mode="constant", value=0)  # left bottom
        x4 = nn.functional.pad(x4, (0, 1, 0, 1), mode="constant", value=0)  # right bottom
        x = tc.cat([x1, x2, x3, x4], dim=1)
        x = F.conv2d(x, self.ADQC_state_gates(self.index), bias=None, stride=1, padding=self.padding)
        return x
