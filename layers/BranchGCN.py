import torch
import torch.nn as nn
import torch.nn.init as init
import math

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature      # (batch_size, 2*num_dims, num_points, k)

class BranchGCN(nn.Module):
    def __init__(self, batch, depth, features, degrees, k=8, support=10, node=1, upsample=False, activation=True):
        self.batch = batch  # bsz
        self.depth = depth  #  layer_id
        self.in_feature = features[depth]  #
        self.out_feature = features[depth+1]
        self.node = node
        self.degree = degrees[depth]  # K: [2,  2,  2,   2,   2,   64]
        self.k = k  # [2,4,8,16,32,2048]: []
        self.upsample = upsample
        self.activation = activation
        super(BranchGCN, self).__init__()

        self.W_root = nn.ModuleList([nn.Linear(features[inx], self.out_feature, bias=False) for inx in range(self.depth+1)])

        if self.upsample:
            self.W_branch = nn.Parameter(torch.FloatTensor(self.node, self.in_feature, self.degree*self.in_feature))

        self.W_loop = nn.Sequential(nn.Linear(self.in_feature, self.in_feature*support, bias=False),
                                    nn.Linear(self.in_feature*support, self.out_feature, bias=False))

        self.bias = nn.Parameter(torch.FloatTensor(1, self.degree, self.out_feature))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.conv1 = nn.Conv2d(6, 64, 1)
        self.conv2 = nn.Conv2d(64, 3, 1)
        self.init_param()

    def init_param(self):
        if self.upsample:
            init.xavier_uniform_(self.W_branch.data, gain=init.calculate_gain('relu'))

        stdv = 1. / math.sqrt(self.out_feature)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, tree):
        root = 0
        for inx in range(self.depth+1):
            root_num = tree[inx].size(1)  # [1,2,4,8,16,32]
            repeat_num = int(self.node / root_num)  # [2,2,2,2,2,64] = [1,2,4,8,16,32] / [1,2,4,8,16,32]
            root_node = self.W_root[inx](tree[inx])  # (bsz, 1, [64,64,...,3]
            root = root + root_node.repeat(1,1,repeat_num).view(self.batch,-1,self.out_feature)

        branch = 0
        if self.upsample:
            branch = tree[-1].unsqueeze(2) @ self.W_branch  # [bsz, self.node, 1, in_feat] -> [bsz, self.node, 1, in_feat*degree]
            branch = self.leaky_relu(branch)
            branch = branch.view(self.batch,self.node*self.degree,self.in_feature)  # [bsz, self.node, 1, in_feat*degree] -> [bsz, self.node*degree, in_feat]

            branch = self.W_loop(branch)  # [bsz, self.node*degree, in_feat] -> [bsz, self.node*degree, in_feat*support] -> [bsz, self.node*degree, out_feat]

            branch = root.repeat(1,1,self.degree).view(self.batch,-1,self.out_feature) + branch
        else:
            branch = self.W_loop(tree[-1])

            branch = root + branch



        if len(tree)==6:

            # branch: bsz, np, 3
            branch = get_graph_feature(branch.permute(0,2,1), k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)

            branch = self.conv1(branch)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
            branch = self.conv2(branch)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
            branch = branch.max(dim=-1, keepdim=False)[0].permute(0,2,1)    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        if self.activation:
            branch = self.leaky_relu(branch + self.bias.repeat(1,self.node,1))

        tree.append(branch)

        return tree