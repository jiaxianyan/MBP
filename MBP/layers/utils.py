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

class MultiHeadAttention(nn.Module):
    def __init__(self, in_feats, num_head, merge):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_head):
            self.heads.append(EdgeWeightAndSum(in_feats))
        self.merge = merge

    def forward(self, g, edge_feats):
        h_g_heads, weight_heads = [], []
        for attn_head in self.heads:
            h_g_head, weigh = attn_head(g, edge_feats)
            h_g_heads.append(h_g_head)
            weight_heads.append(weigh)

        if self.merge == 'concat':
            return torch.cat(h_g_heads, dim=1), torch.cat(weight_heads, dim=1)
        else:
            return torch.mean(torch.stack(h_g_heads)), torch.mean(torch.stack(weight_heads))

class EdgeWeightAndSum_v2(nn.Module):
    """
    for normal use, please delete the 'temporary version' line and meanwhile recover the 'normal version'
    """
    def __init__(self, in_feats):
        super(EdgeWeightAndSum_v2, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.LeakyReLU()
        )

    def forward(self, g, edge_feats):
        with g.local_scope():
            g.edata['e'] = edge_feats
            g.edata['w'] = edge_softmax(g, self.atom_weighting(g.edata['e']))
            weights = g.edata['w']  # temporary version
            h_g_sum = dgl.sum_edges(g, 'e', 'w')
        # return h_g_sum, g.edata['w']  # normal version
        return h_g_sum, weights  # temporary version

class MultiHeadAttention_v2(nn.Module):
    def __init__(self, in_feats, num_head, merge):
        super(MultiHeadAttention_v2, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_head):
            self.heads.append(EdgeWeightAndSum_v2(in_feats))
        self.merge = merge

    def forward(self, g, edge_feats):
        h_g_heads, weight_heads = [], []
        for attn_head in self.heads:
            h_g_head, weigh = attn_head(g, edge_feats)
            h_g_heads.append(h_g_head)
            weight_heads.append(weigh)

        if self.merge == 'concat':
            return torch.cat(h_g_heads, dim=1), torch.cat(weight_heads, dim=1)
        else:
            return torch.mean(torch.stack(h_g_heads)), torch.mean(torch.stack(weight_heads))

class EdgeWeightAndSum_v3(nn.Module):
    """
    for normal use, please delete the 'temporary version' line and meanwhile recover the 'normal version'
    """
    def __init__(self, in_feats):
        super(EdgeWeightAndSum_v3, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.LeakyReLU()
        )

    def forward(self, g, edge_feats):
        with g.local_scope():
            g.edata['e'] = edge_feats
            g.edata['e2'] = self.atom_weighting(g.edata['e'])
            g.edata['w'] = softmax_edges(g, 'e2')
            weights = g.edata['w']  # temporary version
            h_g_sum = dgl.sum_edges(g, 'e', 'w')
        # return h_g_sum, g.edata['w']  # normal version
        return h_g_sum, weights  # temporary version

class MultiHeadAttention_v3(nn.Module):
    def __init__(self, in_feats, num_head, merge):
        super(MultiHeadAttention_v3, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_head):
            self.heads.append(EdgeWeightAndSum_v3(in_feats))
        self.merge = merge

    def forward(self, g, edge_feats):
        h_g_heads, weight_heads = [], []
        for attn_head in self.heads:
            h_g_head, weigh = attn_head(g, edge_feats)
            h_g_heads.append(h_g_head)
            weight_heads.append(weigh)

        if self.merge == 'concat':
            return torch.cat(h_g_heads, dim=1), torch.cat(weight_heads, dim=1)
        else:
            return torch.mean(torch.stack(h_g_heads)), torch.mean(torch.stack(weight_heads))

class ReadsOutLayer(nn.Module):
    """
    for normal use, please delete the 'temporary version' line and meanwhile recover the 'normal version'
    """
    def __init__(self, in_feats, pooling, num_head=None, attn_merge=None):
        super(ReadsOutLayer, self).__init__()
        self.pooling = pooling

        if self.pooling == 'w_sum':
            self.weight_and_sum = EdgeWeightAndSum(in_feats)
        elif self.pooling == 'multi_head':
            self.weight_and_sum = MultiHeadAttention(in_feats, num_head, attn_merge)
        elif self.pooling == 'w_sum_v2':
            self.weight_and_sum = EdgeWeightAndSum_v2(in_feats)
        elif self.pooling == 'multi_head_v2':
            self.weight_and_sum = MultiHeadAttention_v2(in_feats, num_head, attn_merge)
        elif self.pooling == 'w_sum_v3':
            self.weight_and_sum = EdgeWeightAndSum_v3(in_feats)
        elif self.pooling == 'multi_head_v3':
            self.weight_and_sum = MultiHeadAttention_v3(in_feats, num_head, attn_merge)


    def forward(self, bg, edge_feats):

        # h_g_sum, weights = self.weight_and_sum(bg, edge_feats)  # temporary version
        with bg.local_scope():
            bg.edata['e'] = edge_feats
            h_g_max = dgl.max_edges(bg, 'e')
            if self.pooling == 'mean':
                h_p = dgl.mean_edges(bg, 'e')
            elif self.pooling == 'sum':
                h_p = dgl.sum_edges(bg,'e')
            else:
                h_p, weights = self.weight_and_sum(bg, edge_feats)  # normal version
        bg.edata['weights'] = weights
        return torch.cat([h_p, h_g_max], dim=1)  # normal version
