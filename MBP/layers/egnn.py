import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class EGNNConv(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0):
        super(EGNNConv, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the radial feature: ||x_i - x_j||^2
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn
        )

        # \phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, out_size)
        )

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False)
        )

    def message(self, edges):
        """message function for EGNN"""
        # concat features for edge mlp
        if self.edge_feat_size > 0:
            f = torch.cat(
                [edges.src['h'], edges.dst['h'], edges.data['radial'], edges.data['a']],
                dim=-1
            )
        else:
            f = torch.cat([edges.src['h'], edges.dst['h'], edges.data['radial']], dim=-1)

        msg_h = self.edge_mlp(f)
        msg_x = self.coord_mlp(msg_h) * edges.data['x_diff']

        return {'msg_x': msg_x, 'msg_h': msg_h}

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):

        with graph.local_scope():
            # node feature
            graph.ndata['h'] = node_feat
            # coordinate feature
            graph.ndata['x'] = coord_feat
            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata['a'] = edge_feat
            # get coordinate diff & radial features
            graph.apply_edges(fn.u_sub_v('x', 'x', 'x_diff'))
            graph.edata['radial'] = graph.edata['x_diff'].square().sum(dim=1).unsqueeze(-1)
            # normalize coordinate difference
            graph.edata['x_diff'] = graph.edata['x_diff'] / (graph.edata['radial'].sqrt() + 1e-30)
            graph.apply_edges(self.message)
            graph.update_all(fn.copy_e('msg_x', 'm'), fn.mean('m', 'x_neigh'))
            graph.update_all(fn.copy_e('msg_h', 'm'), fn.sum('m', 'h_neigh'))

            h_neigh, x_neigh = graph.ndata['h_neigh'], graph.ndata['x_neigh']

            h = self.node_mlp(
                torch.cat([node_feat, h_neigh], dim=-1)
            )
            x = coord_feat + x_neigh

            return h, x

class EGNN(nn.Module):
    def __init__(self, input_node_dim, input_edge_dim, hidden_dim, num_layers, dropout, JK='sum'):
        super(EGNN, self).__init__()

        self.num_layers = num_layers

        # List of MLPs
        self.egnn_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.egnn_layers.append(EGNNConv(input_node_dim, hidden_dim, hidden_dim, input_edge_dim))
            else:
                self.egnn_layers.append(EGNNConv(hidden_dim, hidden_dim, hidden_dim, input_edge_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.drop = nn.Dropout(dropout)
        self.JK = JK

    def forward(self, g, Perturb=None):
        hidden_rep = []
        node_feats = g.ndata.pop('h').float()
        edge_feats = g.edata['e']
        coord_feats = g.ndata['pos']
        for idx, egnn in enumerate(self.egnn_layers):
            if idx == 0 and Perturb is not  None:
                node_feats = node_feats + Perturb
            node_feats, coord_feats = egnn(g, node_feats, coord_feats, edge_feats)
            node_feats = self.batch_norms[idx](node_feats)
            node_feats = F.relu(node_feats)
            node_feats = self.drop(node_feats)
            hidden_rep.append(node_feats)

        if self.JK == 'sum':
            hidden_rep = [h.unsqueeze(0) for h in hidden_rep]
            return torch.sum(torch.cat(hidden_rep, dim=0), dim=0)
        elif self.JK == 'max':
            hidden_rep = [h.unsqueeze(0) for h in hidden_rep]
            return torch.max(torch.cat(hidden_rep, dim=0), dim=0)[0]
        elif self.JK == 'concat':
            return torch.cat(hidden_rep, dim=1)
        elif self.JK == 'last':
            return hidden_rep[-1]
