import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from .swin_transformer import swin_transformer_tiny, swin_transformer_small, swin_transformer_base
from .resnet import resnet18, resnet50, resnet101
from .graph import create_e_matrix
from .graph_edge_model import GEM
from .basic_block import *

# Gated GCN Used to Learn Multi-dimensional Edge Features and Node Features
class GNNLayer(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate = 0.1):
        super(GNNLayer, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        dim_in = self.in_channels
        dim_out = self.in_channels

        self.U = nn.Linear(dim_in, dim_out, bias=False)
        self.V = nn.Linear(dim_in, dim_out, bias=False)
        self.A = nn.Linear(dim_in, dim_out, bias=False)
        self.B = nn.Linear(dim_in, dim_out, bias=False)
        self.E = nn.Linear(dim_in, dim_out, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)

        self.bnv = nn.BatchNorm1d(num_classes)
        self.bne = nn.BatchNorm1d(num_classes * num_classes)

        self.act = nn.ReLU()

        self.init_weights_linear(dim_in, 1)


    def init_weights_linear(self, dim_in, gain):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        self.U.weight.data.normal_(0, scale)
        self.V.weight.data.normal_(0, scale)
        self.A.weight.data.normal_(0, scale)
        self.B.weight.data.normal_(0, scale)
        self.E.weight.data.normal_(0, scale)


        bn_init(self.bnv)
        bn_init(self.bne)


    def forward(self, x, edge, start, end):

        res = x
        Vix = self.A(x)  # V x d_out
        Vjx = self.B(x)  # V x d_out
        e = self.E(edge)  # E x d_out
        # print(e.shape)
        # print(x.shape)
        # print(start.shape)
        # print(end.shape)

        edge = edge + self.act(self.bne(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec',(start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b,self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)


        Ujx = self.V(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = res + self.act(self.bnv(x))

        return x, edge



# GAT GCN Used to Learn Multi-dimensional Edge Features and Node Features
class GNN(nn.Module):
    def __init__(self, in_channels, num_classes, layer_num = 2):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        start, end = create_e_matrix(self.num_classes)
        self.start = Variable(start, requires_grad=False)
        self.end = Variable(end, requires_grad=False)

        graph_layers = []
        for i in range(layer_num):
            layer = GNNLayer(self.in_channels, self.num_classes)
            graph_layers += [layer]

        self.graph_layers = nn.ModuleList(graph_layers)


    def forward(self, x, edge):
        dev = x.get_device()
        if dev >= 0:
            self.start = self.start.to(dev)
            self.end = self.end.to(dev)
        for i, layer in enumerate(self.graph_layers):
            x, edge = layer(x, edge, self.start, self.end)
        return x, edge



class Head(nn.Module):
    def __init__(self, in_channels, num_main_classes = 27, num_sub_classes = 14):
        super(Head, self).__init__()
        self.in_channels = in_channels
        self.num_main_classes = num_main_classes
        self.num_sub_classes = num_sub_classes

        main_class_linear_layers = []

        for i in range(self.num_main_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            main_class_linear_layers += [layer]
        self.main_class_linears = nn.ModuleList(main_class_linear_layers)

        self.edge_extractor = GEM(self.in_channels, num_main_classes)
        self.gnn = GNN(self.in_channels, num_main_classes, 2)


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

        f_e = self.edge_extractor(f_u, x)
        f_e = f_e.mean(dim=-2)
        f_v, f_e = self.gnn(f_v, f_e)


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
    def __init__(self, num_main_classes = 27, num_sub_classes = 14, backbone='swin_transformer_base'):
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
        self.head = Head(self.out_channels, num_main_classes, num_sub_classes)

    def forward(self, x):
        # x: b d c
        x = self.backbone(x)
        x = self.global_linear(x)
        cl = self.head(x)
        return cl