import torch
from torch import nn
from layer_module import *
from torch.nn.utils import weight_norm
# todo 这个文件就是专门为graphwavenet文件准备的， 不需要的都可以删掉


class calc_adj_5_withLoss_1(nn.Module):
    def __init__(self, node_dim, heads, head_dim, nodes=207, eta=1,
                 gamma=0.001, dropout=0.5, n_clusters=5):
        super(calc_adj_5_withLoss_1, self).__init__()

        self.D = heads * head_dim  # node_dim #
        self.heads = heads
        self.dropout = dropout
        self.eta = eta
        self.gamma = gamma

        self.head_dim = head_dim
        self.node_dim = node_dim
        self.nodes = nodes

        self.query = fc_layer(in_channels=node_dim, out_channels=self.D, need_layer_norm=False)
        self.key = fc_layer(in_channels=node_dim, out_channels=self.D, need_layer_norm=False)
        self.value = fc_layer(in_channels=node_dim, out_channels=self.D, need_layer_norm=False)
        self.mlp = nn.Conv2d(in_channels=self.heads, out_channels=self.heads, kernel_size=(1, 1), bias=True)

        self.bn = nn.LayerNorm(node_dim)
        self.bn1 = nn.LayerNorm(node_dim)
        self.w = nn.Parameter(torch.zeros(size=(nodes, node_dim)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.attn_static = nn.LayerNorm(nodes)
        self.skip_norm = nn.LayerNorm(nodes)
        self.attn_norm = nn.LayerNorm(nodes)
        self.linear_norm = nn.LayerNorm(nodes)
        self.attn_linear = nn.Parameter(torch.zeros(size=(nodes, nodes)))
        nn.init.xavier_uniform_(self.attn_linear.data, gain=1.414)
        self.attn_linear_1 = nn.Parameter(torch.zeros(size=(nodes, nodes)))
        nn.init.xavier_uniform_(self.attn_linear_1.data, gain=1.414)
        self.static_inf_norm = nn.LayerNorm(nodes)
        self.attn_norm_1 = nn.LayerNorm(nodes)
        self.attn_norm_2 = nn.LayerNorm(nodes)
        # ------4-20-----------------添加动态节点模式
        self.dy_model = nn.Parameter(torch.zeros(self.D, nodes))
        nn.init.xavier_uniform_(self.dy_model, gain=1.414)
        self.dy_norm = nn.LayerNorm(node_dim)
        # todo-------4-26------------添加无监督聚类loss
        # self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, node_dim))
        # nn.init.xavier_uniform_(self.cluster_layer.data, gain=1.414)

    def forward(self, nodevec1, nodevec2, input_gc, nodevec_dyStatic, batch_size=64):
        # fixme 发现一个大问题，这里的东西有点不对啊，跟我原来想的是不一样的啊，真的 不一样，
        #  这里的本来的nodevec1不应该被更新的，但是被更新到了啊
        '''
        nodevec1 : 融合输入信息的节点嵌入信息
        nodevec2: 原始的只有节点嵌入信息的节点嵌入
        nodevec_dyStatic: 动态图中的静态信息
        '''
        node_orginal = nodevec2
        nodevec1 = self.bn(nodevec1)
        # nodevec2 = self.bn1(nodevec2)
        #static_graph_inf = self.static_inf_norm(torch.mm(nodevec_dyStatic, nodevec_dyStatic.transpose(1, 0)))
        # resolution_static, resolution_static_t = self.static_graph_1(node_orginal)
        # todo 下面是静态图
        resolution_static = self.static_graph(node_orginal)

        # nodevec1 = nodevec2 =  batch_size, nodes, dim
        # todo --------------------------------------------------------这里动态图 也是会影响静态图更新的啊
        batch_size, nodes, node_dim = batch_size, self.nodes, self.node_dim
        nodevec1_1 = torch.einsum('bnd, nl -> bnl', [nodevec1, self.w]) + nodevec1

        skip_atten = torch.einsum('bnd,bdm->bnm', [nodevec1_1, nodevec1_1.transpose(-1, -2)])
        skip_atten = self.skip_norm(skip_atten)
        # nodevec1 = batch_size, dim, nodes, 1
        nodevec1 = nodevec1.unsqueeze(1).transpose(1, -1)

        # query = batch_size, D, nodes, 1
        query = self.query(nodevec1)
        key = self.key(nodevec1)
        value = self.value(nodevec1)
        # 切分头
        key = key.squeeze(-1).contiguous().view(batch_size, self.heads, self.head_dim, nodes)
        # query = batch_size, heads, nodes, head_dim 64 4 8 40
        query = query.squeeze(-1).contiguous().view(batch_size, self.heads, self.head_dim, nodes).transpose(-1, -2)

        # todo  这里加上dropout试试-------5-17--------------
        key, query = F.dropout(key, 0.5, training=self.training), F.dropout(query, 0.5, training=self.training)

        # attention = batch_size, heads, nodes, nodes 64 4 40 40
        attention = torch.einsum('bhnd, bhdu-> bhnu', [query, key])

        # # -------5-3-----添加--------------------
        attention /= (self.head_dim ** 0.5)
        attention = F.dropout(attention, self.dropout, training=self.training)

        attention = self.mlp(attention) + attention  # batch heads nodes nodes
        # 这里应该是对应上跳过连接这个功能
        resolution = self.attn_norm(torch.sum(attention, dim=1)) + skip_atten #batch node node

        resolution1 = F.relu(torch.einsum('bnm, ml->bnl', [self.linear_norm(resolution), self.attn_linear]))

        # todo 这里也应该添加dropout试试----------5-17------------------
        resolution1 = F.dropout(resolution1, 0.5, training=self.training)

        resolution1 = torch.einsum('bnm, ml -> bnl', [resolution1, self.attn_linear_1])

        relation2 = self.attn_norm_1(resolution1 + resolution)
        #relation2 = self.attn_norm_1(resolution)
        relation2 = F.dropout(relation2, self.dropout, training=self.training)

        # 这里加上这个就是为了防止自注意力崩溃的问题
        # static_graph_inf = static_graph_inf.unsqueeze(0).repeat(batch_size, 1, 1)
        # relation3 = self.attn_norm_2(relation2 + static_graph_inf) #

        # 这里加上动态信息 然后softmax试试
        relation4 = F.softmax(F.relu(relation2), dim=2)
        # relation4 = self.adj_mask(F.relu(relation3), nodes)
        # -------4-8-----------加入图关系损失
        # resolution_static = resolution_static.unsqueeze(0).repeat(batch_size, 1, 1)

        #return relation4, resolution_static
        return relation4
        #, resolution_static_t    # nodevec_dyStatic

    def static_graph(self, nodevec):
        resolution_static = torch.mm(nodevec, nodevec.transpose(1, 0))
        resolution_static = F.softmax(F.relu(self.attn_static(resolution_static)), dim=1)
        return resolution_static

    def graph_loss(self, input, adj, eta=1, gamma=0.001): # 0.0001
        '''
        input:经过编码的输入
        adj:矩阵
        box_num:
        eta:
        '''
        B, N, D = input.shape
        x_i = input.unsqueeze(2).expand(B, N, N, D)
        x_j = input.unsqueeze(1).expand(B, N, N, D)
        box_num = 1 / (N * N)
        dist_loss = eta * torch.norm(x_i - x_j, dim=3) + adj
        dist_loss = torch.exp(dist_loss)
        dist_loss = torch.sum(dist_loss, dim=(1, 2)) * box_num

        f_norm = torch.norm(adj, dim=(1, 2))
        # todo gl_loss
        gl_loss = dist_loss + gamma * f_norm
        # print(dist_loss, f_norm)
        return gl_loss

    def graph_loss_orginal(self, input, adj, eta=1, gamma=0.001):  # 0.0001
        '''
        input:经过编码的输入
        adj:矩阵
        box_num:
        eta:
        '''
        B, N, D = input.shape
        x_i = input.unsqueeze(2).expand(B, N, N, D)
        x_j = input.unsqueeze(1).expand(B, N, N, D)
        dist_loss = torch.pow(torch.norm(x_i - x_j, dim=3), 2) * adj
        # dist_loss = torch.exp(dist_loss)
        dist_loss = torch.sum(dist_loss, dim=(1, 2))

        f_norm = torch.pow(torch.norm(adj, dim=(1, 2)), 2)
        # todo gl_loss
        gl_loss = dist_loss + gamma * f_norm
        # print(dist_loss, '--------------', gamma * f_norm)
        return gl_loss

    def graph_loss_dynamic(self, input, adj, static_adj, gamma=0.001, sentite=0.001, difference=0.01):
        B, N, D = input.shape
        x_i = input.unsqueeze(2).expand(B, N, N, D)
        x_j = input.unsqueeze(1).expand(B, N, N, D)
        dist_loss = torch.pow(torch.norm(x_i - x_j, dim=3), 2) * adj
        # dist_loss = torch.exp(dist_loss)
        dist_loss = torch.sum(dist_loss, dim=(1, 2))

        f_norm = torch.pow(torch.norm(adj, dim=(1, 2)), 2)
        gap = torch.pow(torch.norm(adj - static_adj, dim=2), 2)
        gap = torch.sum(gap)

        gl_loss = sentite * dist_loss + gamma * f_norm + difference * gap
        return gl_loss

    def graph_loss_cluster(self, nodevec=None):
        if nodevec is None:
            return 'need nodevec'
        # 207, 10
        z = nodevec
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()
        p = self.target_distribution(q)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        return kl_loss

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def static_graph_1(self, nodevec):
        resolution_static = torch.mm(nodevec, nodevec.transpose(1, 0))
        resolution = F.relu(self.attn_static(resolution_static))
        resolution_static = F.softmax(resolution, dim=1)
        resolution_static_1 = F.softmax(resolution.transpose(0, 1), dim=1)
        return resolution_static, resolution_static_1


class graph_constructor(nn.Module):
    def __init__(self, nodes, dim, device, time_step, cout=16, heads=2
                 , head_dim=8,
                 eta=1, gamma=0.0001, dropout=0.5, m=0.9, batch_size=64, in_dim=1, is_add1=False):
        super(graph_constructor, self).__init__()
        self.embed1 = nn.Embedding(nodes, dim)
        # -------5-12------自己试试初始化------------
        # for para_uniform in self.embed1.parameters():
        #     reset_embed1_data = para_uniform.data
        # # nn.init.xavier_uniform_(reset_embed1_data, gain=1.414)
        # # nn.init.uniform_(reset_embed1_data)
        # nn.init.kaiming_normal_(reset_embed1_data)

        self.m = m
        self.embed2 = nn.Embedding(nodes, dim)
        for param in self.embed2.parameters():
            param.requires_grad = False
        for para_static, para_w in zip(self.embed2.parameters(), self.embed1.parameters()):
            para_static.data = para_w.data
        self.heads = heads
        self.head_dim = head_dim

        self.out_channel = cout
        self.device = device
        # 这个是词嵌入向量的维度大小
        self.dim = dim
        self.nodes = nodes
        self.time_step = time_step
        if is_add1:
            time_length = time_step + 1
        else:
            time_length = time_step
        self.trans_Merge_line = nn.Conv2d(in_dim, dim, kernel_size=(1, time_length), bias=True) # cout
        self.gate_Fusion_1 = gatedFusion_1(self.dim, device)

        self.calc_adj = calc_adj_5_withLoss_1(node_dim=dim, heads=heads, head_dim=head_dim, nodes=nodes,
                                              eta=eta, gamma=gamma, dropout=dropout)

        self.dim_to_channels = nn.Parameter(torch.zeros(size=(heads * head_dim, cout * time_step)))
        nn.init.xavier_uniform_(self.dim_to_channels.data, gain=1.414)
        self.skip_norm = nn.LayerNorm(time_step)
        self.time_norm = nn.LayerNorm(dim)

    def forward(self, input):
        """
        input = batch_size, 2, nodes ,time_step
        input = batch_size, 1, nodes ,time_step
        """
        batch_size, nodes, time_step = input.shape[0], self.nodes, self.time_step
        for para_static, para_w in zip(self.embed2.parameters(), self.embed1.parameters()):
            para_static.data = para_static.data * self.m + para_w.data * (1 - self.m)


        time_node = input

        time_node = self.trans_Merge_line(time_node)

        time_node = self.time_norm(time_node.squeeze(-1).transpose(1, 2))

        idx = torch.arange(self.nodes).to(self.device)
        nodevec1 = self.embed1(idx)
        nodevec_orginal = nodevec1
        # 下面控制是否使用动量更新
        nodevec2 = self.embed1(idx) #self.embed2(idx) #

        # todo ---------------------修改这里啊---------------------------------

        #nodevec1 = self.gate_Fusion_1(batch_size, nodevec1, time_node) + nodevec1 # nodevec1  time_node# + nodevec1 #

        #adj = self.calc_adj(nodevec1, nodevec_orginal, time_node, nodevec2, batch_size)
        adj = self.calc_adj(time_node, nodevec_orginal, time_node, nodevec2, batch_size)
        return adj


class graph_constructor2(nn.Module):
    def __init__(self, nodes, dim, device, time_step, cout=16, heads=2
                 , head_dim=8,
                 eta=1, gamma=0.0001, dropout=0.5, m=0.9, batch_size=64, in_dim=1, is_add1=False):
        super(graph_constructor2, self).__init__()
        self.embed1 = nn.Embedding(nodes, dim)

        self.m = m
        self.embed2 = nn.Embedding(nodes, dim)
        for param in self.embed2.parameters():
            param.requires_grad = False
        for para_static, para_w in zip(self.embed2.parameters(), self.embed1.parameters()):
            para_static.data = para_w.data
        self.heads = heads
        self.head_dim = head_dim

        self.out_channel = cout
        self.device = device
        # 这个是词嵌入向量的维度大小
        self.dim = dim
        self.nodes = nodes
        self.time_step = time_step
        if is_add1:
            time_length = time_step + 1
        else:
            time_length = time_step
        self.trans_Merge_line = nn.Conv2d(in_dim, dim, kernel_size=(1, time_length), bias=True) # cout
        self.gate_Fusion_1 = gatedFusion_1(self.dim, device)


        self.dim_to_channels = nn.Parameter(torch.zeros(size=(heads * head_dim, cout * time_step)))
        nn.init.xavier_uniform_(self.dim_to_channels.data, gain=1.414)
        self.skip_norm = nn.LayerNorm(time_step)
        self.time_norm = nn.LayerNorm(dim)
        self.bn = nn.LayerNorm(dim)

    def forward(self, input):
        """
        input = batch_size, 2, nodes ,time_step
        input = batch_size, 1, nodes ,time_step
        """
        batch_size, nodes, time_step = input.shape[0], self.nodes, self.time_step
        for para_static, para_w in zip(self.embed2.parameters(), self.embed1.parameters()):
            para_static.data = para_static.data * self.m + para_w.data * (1 - self.m)


        time_node = input

        time_node = self.trans_Merge_line(time_node)

        time_node = self.time_norm(time_node.squeeze(-1).transpose(1, 2))

        idx = torch.arange(self.nodes).to(self.device)
        nodevec1 = self.embed1(idx)

        nodevec1 = self.gate_Fusion_1(batch_size, nodevec1, time_node) + nodevec1 # nodevec1  time_node# + nodevec1 #
        ####x
        nodevec1 = self.bn(nodevec1)
        dynamic_em = torch.bmm(nodevec1, nodevec1.transpose(2, 1))
        adj_em = F.softmax(F.relu(dynamic_em), dim=2)
        
        return adj_em #adj