import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax

class intra_message(nn.Module):
    def __init__(self,node_feat_size, graph_feat_size, dropout):
        super(intra_message, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )

        self.bn_layer = nn.BatchNorm1d(graph_feat_size)

    def apply_edges(self, edges):
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self,g, node_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])
        g.edata['a'] = edge_softmax(g, logits)
        g.ndata['hv'] = self.project_node(node_feats)
        g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))

        return F.elu(g.ndata['c'])

class inter_message(nn.Module):
    def __init__(self,in_dim, out_dim, dropout):
        super(inter_message, self).__init__()
        self.project_edges = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU()
        )
    def apply_edges(self, edges):
        return {'m': self.project_edges(torch.cat([edges.data['e'],edges.src['h'], edges.dst['h']], dim=1))}

    def forward(self,g, node_feats):
        g = g.local_var()
        g.ndata['h'] = node_feats
        g.update_all(self.apply_edges, fn.mean('m','c'))
        return F.elu(g.ndata['c'])

class update_node_feats(nn.Module):
    def __init__(self,in_dim, out_dim, dropout):
        super(update_node_feats, self).__init__()
        self.gru = nn.GRUCell(out_dim, out_dim)
        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU()
        )
        self.bn_layer = nn.BatchNorm1d(out_dim)

    def forward(self, g, node_feats, intra_m, inter_m):
        g = g.local_var()
        return self.bn_layer(F.relu(self.gru(self.project_node(torch.cat([node_feats, intra_m, inter_m], dim=1)),node_feats)))


