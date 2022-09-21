import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.BranchGCN import BranchGCN, knn, get_graph_feature
from torch.nn.utils import spectral_norm
from torch.autograd import Variable
import numpy as np


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = spectral_norm(torch.nn.Conv1d(3, 64, 1))
        self.conv2 = spectral_norm(torch.nn.Conv1d(64, 128, 1))
        self.conv3 = spectral_norm(torch.nn.Conv1d(128, 1024, 1))
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()


    def forward(self, x):
        # x: (bsz, np, 3)
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class Discriminator_with_Classifier(nn.Module):
    def __init__(self, features, k=16):
        super(Discriminator_with_Classifier, self).__init__()

        self.stn = STN3d()
        self.conv1 = spectral_norm(nn.Conv1d(3, 64, 1))
        self.conv2 = spectral_norm(nn.Conv1d(64, 128, 1))
        self.conv3 = spectral_norm(nn.Conv1d(128, 1024, 1))
        self.fc1 = spectral_norm(nn.Linear(1024, 512))
        self.fc2 = spectral_norm(nn.Linear(512, 256))
        self.fc3 = nn.Linear(256, k)
        self.relu = nn.ReLU()

        self.layer_num = len(features)-1

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(features[inx], features[inx+1], kernel_size=1, stride=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.final_layer = nn.Sequential(nn.Linear(features[-1], features[-3]),
                                         nn.Linear(features[-3], features[-5]),
                                         nn.Linear(features[-5], 1))
        # self.act = nn.Sigmoid()

    def forward(self, f):
        # f: (bsz, 3, np)
        # feat: (bsz, np, 3)
        f = f.permute(0,2,1)  # f: (bsz, np, 3)
        feat = f
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)

        # out = F.avg_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)

        out = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        # score = self.act(self.final_layer(out)) # (B, 1)
        score = self.final_layer(out)


        trans = self.stn(f)  # bsz, 3, 3
        x = f.permute(0,2,1)  # f: bsz, 3, np
        x = torch.bmm(x, trans)  # x: bsz, np, 3
        x = x.transpose(2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x1 = x.view(-1, 1024)

        x2 = F.relu(self.fc1(x1))
        x3 = F.relu(self.fc2(x2))
        x4 = self.fc3(x3)
        # actv = torch.cat((x1, x2, x3, x4), dim=1)

        return score, x4  # score: bsz, 1; x4: bsz, 16


class Discriminator(nn.Module):
    def __init__(self, features):
        super(Discriminator, self).__init__()

        self.layer_num = len(features)-1

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(features[inx], features[inx+1], kernel_size=1, stride=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.final_layer = nn.Sequential(nn.Linear(features[-1], features[-3]),
                                         nn.Linear(features[-3], features[-5]),
                                         nn.Linear(features[-5], 1))
        # self.act = nn.Sigmoid()

    def forward(self, f):
        # f: (bsz, 3, np)
        # feat: (bsz, np, 3)
        f = f.permute(0,2,1)  # f: (bsz, np, 3)
        feat = f
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)

        # out = F.avg_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1) # mine
        out = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        # score = self.act(self.final_layer(out)) # (B, 1)
        score = self.final_layer(out)


        return score  # score: bsz, 1; x4: bsz, 16

class DGCNN_Transformer(nn.Module):
    def __init__(self, num_classes=16, k=20, hidden_dim=1024, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.stn = STN3d()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(hidden_dim)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, hidden_dim, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, 3, kernel_size=1, bias=False)

    def forward(self, x, l):
        # x: (bsz, 3, np)
        l = F.one_hot(l, num_classes=self.num_classes)
        x = x.permute(0,2,1)
        batch_size = x.size(0)
        num_points = x.size(2)

        trans = self.stn(x)  # bsz, 3, 3
        x = x.permute(0,2,1)  # f: bsz, 3, np
        x = torch.bmm(x, trans)  # x: bsz, np, 3
        x = x.transpose(2,1)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1).float()   # (batch_size, num_categoties, 1)
        l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)                      # (batch_size, 256, num_points) -> (batch_size, 3, num_points)
        x = x.permute(0,2,1)
        return torch.tanh(x)

class Generator(nn.Module):
    def __init__(self, batch_size, features, degrees, support):
        self.batch_size = batch_size
        self.layer_num = len(features)-1
        assert self.layer_num == len(degrees), "Number of features should be one more than number of degrees."
        self.pointcloud = None
        super(Generator, self).__init__()

        vertex_num = 1
        self.gcn = nn.Sequential()
        for inx in range(self.layer_num):
            if inx == self.layer_num-1:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    BranchGCN(batch=self.batch_size, depth=inx, features=features, degrees=degrees, support=support, node=vertex_num, upsample=True, activation=False))
            else:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    BranchGCN(batch=self.batch_size, depth=inx, features=features, degrees=degrees, support=support, node=vertex_num, upsample=True, activation=True))
            vertex_num = int(vertex_num * degrees[inx])
        self.dgcnn = DGCNN_Transformer()


    def forward(self, tree, l):
        self.label = l
        feat = self.gcn(tree)
        self.pre_pc = feat[-1]

        self.pointcloud = self.dgcnn(feat[-1], l)

        return self.pre_pc, self.pointcloud

    def modify(self, prePointCloud, l):
        return self.dgcnn(prePointCloud, l)

    def getPointcloud(self):
        return self.pointcloud[-1]

    def getPrePointCloud(self):
        return self.pre_pc[-1]

    def getLabel(self):
        return self.label[-1]


if __name__ == "__main__":

    l = torch.randint(0, 16,(12,1))
    z = torch.rand([12, 1, 96])

    gen = Generator(batch_size=12, features=[96, 64, 64,  64,  64,  64, 3], degrees=[2,  2,  2,   2,   2,   64], support=10)

    pre_x, gen_x = gen([z], l)  # pre_x: (bsz, np, 3), gen_x: (bsz, np, 3)


    print("done")