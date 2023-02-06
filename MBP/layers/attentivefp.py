import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax

class AttentiveGRU1(nn.Module):

    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, edge_feats, node_feats):
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class AttentiveGRU2(nn.Module):
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, node_feats):
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class GetContext(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def apply_edges1(self, edges):
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])


class GNNLayer(nn.Module):
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)
        self.bn_layer = nn.BatchNorm1d(graph_feat_size)

    def apply_edges(self, edges):
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        return self.bn_layer(self.attentive_gru(g, logits, node_feats))

class ModifiedAttentiveFPGNNV2(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.,
                 jk='sum'):
        super(ModifiedAttentiveFPGNNV2, self).__init__()
        self.jk = jk
        self.graph_feat_size = graph_feat_size
        self.num_layers = num_layers
        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

    def forward(self, g):
        atom_feats = g.ndata.pop('h').float()
        bond_feats = g.edata.pop('e')
        node_feats = self.init_context(g, atom_feats, bond_feats)
        h_list = [node_feats]
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
            h_list.append(node_feats)

        if self.jk=='sum':
            h_list = [h.unsqueeze(0) for h in h_list]
            return torch.sum(torch.cat(h_list, dim=0), dim=0)
        elif self.jk=='max':
            h_list = [h.unsqueeze(0) for h in h_list]
            return torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.jk=='concat':
            return torch.cat(h_list, dim = 1)
        elif self.jk=='last':
            return h_list[-1]


class DTIConvGraph3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DTIConvGraph3, self).__init__()
        # the MPL for update the edge state
        self.mpl = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU())

    def EdgeUpdate(self, edges):
        return {'e': self.mpl(torch.cat([edges.data['e'],edges.src['h'], edges.dst['h']], dim=1))}

    def forward(self, bg):
        with bg.local_scope():
            bg.apply_edges(self.EdgeUpdate)
            return bg.edata['e']

class DTIConvGraph3Layer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):  # in_dim = graph module1 output dim + 1
        super(DTIConvGraph3Layer, self).__init__()
        # the MPL for update the edge state
        self.grah_conv = DTIConvGraph3(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(out_dim)

    def forward(self, bg):
        new_feats = self.grah_conv(bg)
        return self.bn_layer(self.dropout(new_feats))


class DTIConvGraph3_IGN_basic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DTIConvGraph3_IGN_basic, self).__init__()
        # the MPL for update the edge state
        self.mpl = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU())

    def EdgeUpdate(self, edges):
        return {'e': self.mpl(torch.cat([edges.data['e'], edges.src['h'] + edges.dst['h']], dim=1))}

    def forward(self, bg):
        with bg.local_scope():
            bg.apply_edges(self.EdgeUpdate)
            return bg.edata['e']

class DTIConvGraph3Layer_IGN_basic(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):  # in_dim = graph module1 output dim + 1
        super(DTIConvGraph3Layer_IGN_basic, self).__init__()
        # the MPL for update the edge state
        self.grah_conv = DTIConvGraph3_IGN_basic(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(out_dim)

    def forward(self, bg):
        new_feats = self.grah_conv(bg)
        return self.bn_layer(self.dropout(new_feats))


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
            # weights = g.edata['w']  # temporary version
            h_g_sum = dgl.sum_edges(g, 'e', 'w')
        return h_g_sum  # normal version
        # return h_g_sum, weights  # temporary version

class ReadsOutLayer(nn.Module):
    """
    for normal use, please delete the 'temporary version' line and meanwhile recover the 'normal version'
    """
    def __init__(self, in_feats, pooling):
        super(ReadsOutLayer, self).__init__()
        self.weight_and_sum = EdgeWeightAndSum(in_feats)
        self.pooling = pooling

    def forward(self, bg, edge_feats):

        if self.pooling == 'w_sum':
            h_g_sum_w = self.weight_and_sum(bg, edge_feats)  # normal version
            # return h_g_sum_w

        # h_g_sum, weights = self.weight_and_sum(bg, edge_feats)  # temporary version
        with bg.local_scope():
            bg.edata['e'] = edge_feats
            h_g_max = dgl.max_edges(bg, 'e')
            if self.pooling == 'mean':
                h_g_mean = dgl.mean_edges(bg, 'e')
                # return h_g_mean
            elif self.pooling == 'sum':
                h_g_sum = dgl.sum_edges(bg,'e')
                # return h_g_sum
        if self.pooling == 'mean':
            h_g = torch.cat([h_g_mean, h_g_max], dim=1)
        elif self.pooling == 'sum':
            h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        elif self.pooling == 'w_sum':
            h_g = torch.cat([h_g_sum_w, h_g_max], dim=1)
        #
        return h_g  # normal version
