import torch.nn as nn
from tqdm import tqdm
import torch as th
import pdb
import torch.nn.functional as F
import torch
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.nn import GraphConv, GATConv, SGConv, APPNPConv
from models.layers import DGCNLayer
import pdb

class GCNModel(nn.Module):
    def __init__(self, args, in_size, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GraphConv(in_size, args.h_feats)
        self.conv2 = GraphConv(args.h_feats, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h

class GATModel(nn.Module):
    def __init__(self, args, in_size, num_classes):
        super().__init__()
        heads = [args.num_heads, 1]
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.gat_layers.append(
            dglnn.GATConv(
                in_size,
                args.h_feats,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        self.gat_layers.append(
            dglnn.GATConv(
                args.h_feats * heads[0],
                num_classes,
                heads[1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
            )
        )

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h

class SGCModel(nn.Module):
    def __init__(self, args, in_size, num_classes):
        super().__init__()
        self.conv = SGConv(in_size, num_classes, k = 2)

    def forward(self, g, in_feat):
        h = self.conv(g, in_feat)
        return h

class APPNPModel(nn.Module):
    def __init__(
        self,
        args,
        in_size,
        num_classes,
    ):
        super(APPNPModel, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_size, args.h_feats))
        # hidden layers
        # for i in range(1, len(hiddens)):
        #     self.layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))
        # output layer
        self.layers.append(nn.Linear(args.h_feats, num_classes))
        self.activation = F.relu
        if args.in_drop:
            self.feat_drop = nn.Dropout(args.in_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(args.k, args.alpha, args.edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(g, h)
        return h

class DGCN(nn.Module):
    def __init__(self, args, in_size, num_classes):
        super(DGCN, self).__init__()
        self.conv1 = DGCNLayer(in_size, args.h_feats, args.alpha)
        self.conv2 = DGCNLayer(args.h_feats, num_classes, args.alpha)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h
