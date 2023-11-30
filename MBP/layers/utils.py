import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax
from dgl import softmax_edges

class FC(nn.Module):
    def __init__(self, d_graph_layer, fc_hidden_dim, dropout, n_tasks):
        super(FC, self).__init__()

        self.predict = nn.ModuleList()
        for index,dim in enumerate(fc_hidden_dim):
            self.predict.append(nn.Linear(d_graph_layer, dim))
            self.predict.append(nn.Dropout(dropout))
            self.predict.append(nn.LeakyReLU())
            self.predict.append(nn.BatchNorm1d(dim))
            d_graph_layer = dim
        self.predict.append(nn.Linear(d_graph_layer, n_tasks))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        # return torch.sigmoid(h)
        return h

class EdgeWeightAndSum(nn.Module):
    """
    for normal use, please delete the 'temporary version' line and meanwhile recover the 'normal version'
    """
    def __init__(self, in_feats):
        super(EdgeWeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Tanh()
        )

    def forward(self, g, edge_feats):
        with g.local_scope():
            g.edata['e'] = edge_feats
            g.edata['w'] = self.atom_weighting(g.edata['e'])
            weights = g.edata['w']  # temporary version
            h_g_sum = dgl.sum_edges(g, 'e', 'w')
        # return h_g_sum, g.edata['w']  # normal version
        return h_g_sum, weights  # temporary version

class ReadsOutLayer(nn.Module):
    """
    for normal use, please delete the 'temporary version' line and meanwhile recover the 'normal version'
    """
    def __init__(self, in_feats, pooling):
        super(ReadsOutLayer, self).__init__()
        self.pooling = pooling
        self.weight_and_sum = EdgeWeightAndSum(in_feats)

    def forward(self, bg, edge_feats):
        # h_g_sum, weights = self.weight_and_sum(bg, edge_feats)  # temporary version
        with bg.local_scope():
            bg.edata['e'] = edge_feats
            h_g_max = dgl.max_edges(bg, 'e')
            h_p, weights = self.weight_and_sum(bg, edge_feats)  # normal version
        bg.edata['weights'] = weights

        return torch.cat([h_p, h_g_max], dim=1)  # normal version
