import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from .swin_transformer import swin_transformer_tiny, swin_transformer_small, swin_transformer_base
from .resnet import resnet18, resnet50, resnet101
from .graph import normalize_digraph
from .basic_block import *


class GNN(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(GNN, self).__init__()
        # in_channels: dim of node feature
        # num_classes: num of nodes
        # neighbor_num: K in paper and we select the top-K nearest neighbors for each node feature.
        # metric: metric for assessing node similarity. Used in FGG module to build a dynamical graph
        # X' = ReLU(X + BN(V(X) + A x U(X)) )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        # network
        self.U = nn.Linear(self.in_channels,self.in_channels)
        self.V = nn.Linear(self.in_channels,self.in_channels)
        self.bnv = nn.BatchNorm1d(num_classes)

        # init
        self.U.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.bnv.weight.data.fill_(1)
        self.bnv.bias.data.zero_()

    def forward(self, x):
        b, n, c = x.shape

        # build dynamical graph
        if self.metric == 'dots':
            si = x.detach()
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'cosine':
            si = x.detach()
            si = F.normalize(si, p=2, dim=-1)
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'l1':
            si = x.detach().repeat(1, n, 1).view(b, n, n, c)
            si = torch.pow(si.transpose(1, 2) - si,2)
            si = torch.sqrt(si.sum(dim=-1))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
            adj = (si <= threshold).float()

        elif self.metric == 'l2':
            si = x.detach().repeat(1, n, 1).view(b, n, n, c)
            si = torch.abs(si.transpose(1, 2) - si)
            si = si.sum(dim=-1)
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
            adj = (si <= threshold).float()

        else:
            raise Exception("Error: wrong metric: ", self.metric)

        # GNN process
        A = normalize_digraph(adj)
        aggregate = torch.einsum('b i j, b j k->b i k', A, self.V(x))
        x = self.relu(x + self.bnv(aggregate + self.U(x)))
        return x


class Head(nn.Module):
    def __init__(self, in_channels, num_main_classes = 27, num_sub_classes = 14, neighbor_num=4, metric='dots'):
        super(Head, self).__init__()
        self.in_channels = in_channels
        self.num_main_classes = num_main_classes
        self.num_sub_classes = num_sub_classes

        main_class_linear_layers = []

        for i in range(self.num_main_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            main_class_linear_layers += [layer]
        self.main_class_linears = nn.ModuleList(main_class_linear_layers)

        self.gnn = GNN(self.in_channels, self.num_main_classes,neighbor_num=neighbor_num,metric=metric)
        self.main_sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_main_classes, self.in_channels)))

        self.sub_sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_sub_classes, self.in_channels)))
        self.sub_list = [0,1,2,4,7,8,11]

        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.main_sc)
        nn.init.xavier_uniform_(self.sub_sc)

    def forward(self, x):
        # AFG
        f_u = []
        for i, layer in enumerate(self.main_class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)
        # FGG
        f_v = self.gnn(f_v)
        b, n, c = f_v.shape

        main_sc = self.main_sc
        main_sc = self.relu(main_sc)
        main_sc = F.normalize(main_sc, p=2, dim=-1)
        main_cl = F.normalize(f_v, p=2, dim=-1)
        main_cl = (main_cl * main_sc.view(1, n, c)).sum(dim=-1)

        sub_cl = []
        for i, index in enumerate(self.sub_list):
            au_l = 2*i
            au_r = 2*i + 1
            main_au = F.normalize(f_v[:, index], p=2, dim=-1)

            sc_l = F.normalize(self.relu(self.sub_sc[au_l]), p=2, dim=-1)
            sc_r = F.normalize(self.relu(self.sub_sc[au_r]), p=2, dim=-1)

            cl_l = (main_au * sc_l.view(1, c)).sum(dim=-1)
            cl_r = (main_au * sc_r.view(1, c)).sum(dim=-1)
            sub_cl.append(cl_l[:,None])
            sub_cl.append(cl_r[:,None])
        sub_cl = torch.cat(sub_cl, dim=-1)
        cl = torch.cat([main_cl, sub_cl], dim=-1)
        return cl


class MEFARG(nn.Module):
    def __init__(self, num_main_classes = 27, num_sub_classes = 14, backbone='swin_transformer_base', neighbor_num=4, metric='dots'):
        super(MEFARG, self).__init__()
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels // 2
            self.backbone.head = None
        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18()
            elif backbone == 'resnet101':
                self.backbone = resnet101()
            else:
                self.backbone = resnet50()
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.out_channels = self.in_channels // 4
            self.backbone.fc = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = Head(self.out_channels, num_main_classes, num_sub_classes, neighbor_num, metric)

    def forward(self, x):
        # x: b d c
        x = self.backbone(x)
        x = self.global_linear(x)
        cl = self.head(x)
        return cl
