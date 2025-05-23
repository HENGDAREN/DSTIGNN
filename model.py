import torch
import torch.nn as nn
from torch.nn import Conv2d, Parameter
import torch.nn.functional as F

from graph_constuct_fgwn_2 import graph_constructor,graph_constructor2
from RevIN import RevIN


#这个是增加的动态gcn的三个类
class dnconv(nn.Module):
    def __init__(self):
        super(dnconv, self).__init__()

    def forward(self, x, A):
        if len(A.size()) == 2:
            A = A.unsqueeze(0).repeat(x.shape[0], 1, 1)
        #A = torch.tensor(A)
        # x = torch.einsum('nvw, ncvl->ncwl', [A, x])
        #x = torch.einsum('nvw, ncwl->ncvl', [A, x])
        x = torch.einsum('ncwl, nvw->ncvl', [x, A])
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class Diffusion_GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2): #order = 2
        super(Diffusion_GCN, self).__init__()
        # self.nconv = nconv()
        self.nconv = dnconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order
        # --------------------------修改----4-12----------------------------
        # self.weights =

    def forward(self, x, support):
        out = [x]
        x1 = self.nconv(x, support)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1, support)
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
######


class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, :, :, ::2]

    def odd(self, x):
        return x[:, :, :, 1::2]

    def forward(self, x):
        return (self.even(x), self.odd(x))


class IDGCN(nn.Module):
    def __init__(self, device, channels, splitting=True, num_nodes=170, dropout=0.25, pre_adj=None, pre_adj_len=1):
        super(IDGCN, self).__init__()

        device = device
        self.dropout = dropout
        self.pre_adj_len = pre_adj_len
        self.num_nodes = num_nodes
        self.splitting = splitting
        self.pre_graph = pre_adj or []
        self.split = Splitting()

        Conv1 = []
        Conv2 = []
        Conv3 = []
        Conv4 = []
        pad_l = 3
        pad_r = 3

        apt_size = 10
        aptinit = pre_adj[0]
        self.pre_adj_len = 1

        Conv1 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh(),
        ]


        Conv2 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh()
        ]

        Conv4 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh()
        ]
        Conv3 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh()
        ]

        self.conv1 = nn.Sequential(*Conv1)
        self.conv2 = nn.Sequential(*Conv2)
        self.conv3 = nn.Sequential(*Conv3)
        self.conv4 = nn.Sequential(*Conv4)

        self.diffusion_conv = Diffusion_GCN(
            channels, channels, dropout, support_len=self.pre_adj_len)


    def forward(self, x, adj):
        if self.splitting:
            (x_even, x_odd) = self.split(x)   #64 64 170 6
        else:
            (x_even, x_odd) = x


        x1 = self.conv1(x_even) #64 64 170 6

        x1 = x1+self.diffusion_conv(x1, dadj)
        d = x_odd.mul(torch.tanh(x1))

        x2 = self.conv2(x_odd)

        x2 = x2+self.diffusion_conv(x2, dadj)
        c = x_even.mul(torch.tanh(x2))

        x3 = self.conv3(c)

        x3 = x3+self.diffusion_conv(x3, dadj)
        x_odd_update = d - x3 # Either "+" or "-" here does not have much effect on the results.

        x4 = self.conv4(d)

        x4 = x4+self.diffusion_conv(x4, dadj)
        x_even_update = c + x4 # Either "+" or "-" here does not have much effect on the results.

        return (x_even_update, x_odd_update, dadj)


class IDGCN_Tree(nn.Module):
    def __init__(self, device, num_nodes, channels, num_levels, dropout, pre_adj=None, pre_adj_len=1):
        super().__init__()
        self.levels = num_levels
        self.pre_graph = pre_adj or []

        self.IDGCN1 = IDGCN(splitting=True, channels=channels, device=device,
                            pre_adj=pre_adj, num_nodes=num_nodes, dropout=dropout, pre_adj_len=pre_adj_len)
        self.IDGCN2 = IDGCN(splitting=True, channels=channels, device=device,
                            pre_adj=pre_adj, num_nodes=num_nodes, dropout=dropout, pre_adj_len=pre_adj_len)
        self.IDGCN3 = IDGCN(splitting=True, channels=channels, device=device,
                            pre_adj=pre_adj, num_nodes=num_nodes, dropout=dropout, pre_adj_len=pre_adj_len)

        self.a = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)
        self.b = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)
        self.c = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)

    def concat(self, even, odd):
        even = even.permute(3, 1, 2, 0)
        odd = odd.permute(3, 1, 2, 0)
        len = even.shape[0]
        _ = []
        for i in range(len):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        return torch.cat(_, 0).permute(3, 1, 2, 0)

    def forward(self, x, adj):
        # x_even_update1, x_odd_update1, dadj1 = self.IDGCN1(x)
        # x_even_update2, x_odd_update2, dadj2 = self.IDGCN2(x_even_update1)
        # x_even_update3, x_odd_update3, dadj3 = self.IDGCN3(x_odd_update1)

        x_even_update1, x_odd_update1, dadj1 = self.IDGCN1(x,adj)
        x_even_update2, x_odd_update2, dadj2 = self.IDGCN2(x_even_update1,adj)
        x_even_update3, x_odd_update3, dadj3 = self.IDGCN3(x_odd_update1,adj)
        concat1 = self.concat(x_even_update2, x_odd_update2)
        concat2 = self.concat(x_even_update3, x_odd_update3)
        concat0 = self.concat(concat1, concat2)
        adj = dadj1*self.a+dadj2*self.b+dadj3*self.c
        return concat0, adj


class STIDGCN(nn.Module):
    def __init__(self, device, num_nodes, channels, dropout=0.25, pre_adj=None):
        super().__init__()

        self.dropout = dropout
        self.num_nodes = num_nodes
        self.input_len = 12
        self.output_len = 12
        self.num_levels = 2
        self.groups = 1
        input_channel = 1
        apt_size = 10

        ##新增加的参数
        self.gl_bool = True
        ##
        self.pre_graph = pre_adj or []
        self.pre_adj_len = len(self.pre_graph)+1

        # aptinit = pre_adj[0]
        # nodevecs = self.svd_init(apt_size, aptinit)
        
        #self.nodevec1 = nn.Parameter(torch.randn(num_nodes, apt_size), requires_grad=True)
        #self.nodevec2 = nn.Parameter(torch.randn(apt_size, num_nodes), requires_grad=True)
        # self.nodevec1, self.nodevec2 = [
        #      Parameter(n.to(device), requires_grad=True) for n in nodevecs]

        # self.a = nn.Parameter(torch.rand(1).to(
        #     device=device), requires_grad=True).to(device)

        self.start_conv = nn.Conv2d(in_channels=input_channel,
                                    out_channels=channels,
                                    kernel_size=(1, 1))

        self.tree = IDGCN_Tree(
            device=device,
            channels=channels,
            num_nodes=self.num_nodes,
            num_levels=self.num_levels,
            dropout=self.dropout,
            pre_adj_len=self.pre_adj_len,
            pre_adj=pre_adj
        )

        # self.diffusion_conv = Diffusion_GCN(
        #     channels, channels, dropout, support_len=self.pre_adj_len)

        self.Conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=256,
                               kernel_size=(1, 1), stride=(1, 1))
        self.Conv2 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=(1, 12), stride=(1, 1))
        self.Conv3 = nn.Conv2d(in_channels=512,
                               out_channels=12,
                               kernel_size=(1, 1), stride=(1, 1))
        #开始增加图学习
        # self.graph_construct = graph_constructor(num_nodes, apt_size, device, self.input_len, eta=1, in_dim=1,
        #                                          gamma=0.001, dropout=0.5, m=0.9, batch_size=32)
        self.graph_construct = graph_constructor2(num_nodes, apt_size, device, self.input_len, eta=1, in_dim=1,
                                                 gamma=0.001, dropout=0.5, m=0.9, batch_size=32)
        self.depth = 1


    def forward(self, input):
        x = input #64 1 170 12

        x = self.start_conv(x) #64 64 170 12
        ##这里是新加的
        if self.gl_bool:
            #dynamic_adj, static_adj = self.graph_construct(input)
            dynamic_adj = self.graph_construct(input)
        ####
        # adaptive_adj = F.softmax(
        #      F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        skip = x
        #x, dadj = self.tree(x, adaptive_adj)

        x, dadj = self.tree(x, dynamic_adj)

        x = skip + x

        for i in range(self.depth-1):
            skip = x
            x, dadj = self.tree(x, dynamic_adj)
            x = skip + x


        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = self.Conv3(x)   #64 12 170 1
        #新增加的
    
        return x
