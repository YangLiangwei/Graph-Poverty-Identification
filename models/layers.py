import torch.nn as nn
import torch as th
import pdb
from tqdm import tqdm
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn as dglnn
import numpy as np
from fast_pytorch_kmeans import KMeans


class GCNLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.k = args.k
        self.dim = args.embed_size
        self.weight = th.nn.Parameter(th.randn(self.dim, self.dim))
        self.bias = th.nn.Parameter(th.zeros(self.dim))
        th.nn.init.xavier_uniform_(self.weight)

    def reduction(self, nodes):
        # -1 indicate user-> node, which does not include category information
        mail = nodes.mailbox['m']
        batch_size, neighbor_size, feature_size = mail.shape

        if nodes.mailbox['m'].shape[1] <= self.k:
            mail = mail.sum(dim = 1)
        else:
            weight = th.ones(batch_size, neighbor_size, device = mail.device)
            selected = th.multinomial(weight, self.k)
            mail = mail[th.arange(batch_size, dtype = th.long, device = mail.device).unsqueeze(-1), selected]
            mail = mail.sum(dim = 1)
        return {'h': mail}


    def forward(self, graph, h, etype):
        with graph.local_scope():
            src, _ , dst = etype
            feat_src = h[src]
            feat_dst = h[dst]
            aggregate_fn = fn.copy_src('h', 'm')

            degs = graph.out_degrees(etype = etype).float().clamp(min = 1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = th.reshape(norm, shp)
            feat_src = feat_src * norm

            graph.nodes[src].data['h'] = feat_src
            graph.update_all(aggregate_fn, self.reduction, etype = etype)

            # rst = graph.dstdata['h'][dst]
            rst = graph.nodes[dst].data['h']
            rst = th.matmul(rst, self.weight)
            degs = graph.in_degrees(etype = etype).float().clamp(min = 1)
            degs = th.clamp(degs, 0, self.k)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm
            rst += self.bias
            return rst

class DGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, alpha):
        super().__init__()
        self.weight = th.nn.Parameter(th.randn(in_feat, out_feat))
        self.bias = th.nn.Parameter(th.zeros(out_feat))
        self.alpha = th.tensor(alpha)

        th.nn.init.xavier_uniform_(self.weight)


    def aggregation(self, edges):
        return {'m': edges.src['h'], 'd': edges.data['distance']}

    def reduction(self, nodes):
        mail = nodes.mailbox['m']
        distance = nodes.mailbox['d'].type(th.float32)
        weight = th.pow(self.alpha, distance)
        mail = mail * weight.unsqueeze(-1)
        return {'h': mail.mean(1)}


    def forward(self, graph, h):
        with graph.local_scope():

            feat_src = h
            aggregate_fn = fn.copy_src('h', 'm')
            degs = graph.out_degrees()
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = th.reshape(norm, shp)
            feat_src = feat_src * norm

            graph.ndata['h'] = feat_src
            graph.update_all(self.aggregation, self.reduction)

            feat_dst = graph.ndata['h']
            rst = graph.ndata['h']
            rst = th.matmul(rst, self.weight)
            degs = graph.in_degrees()
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm
            rst += self.bias
            return rst
