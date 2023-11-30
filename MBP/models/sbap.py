import torch
import torch.nn as nn
from MBP import layers, losses

class GNNs(nn.Module):
    def __init__(self, nLigNode, nLigEdge, nLayer, nHid, JK, GNN):
        super(GNNs, self).__init__()
        if GNN == 'GCN':
            self.Encoder = layers.GCN(nLigNode, hidden_feats=[nHid] * nLayer)
        elif GNN == 'GAT':
            self.Encoder = layers.GAT(nLigNode, hidden_feats=[nHid] * nLayer)
        elif GNN == 'GIN':
            self.Encoder = layers.GIN(nLigNode, nHid, nLayer, num_mlp_layers=2, dropout=0.1, learn_eps=False,
                               neighbor_pooling_type='sum', JK=JK)
        elif GNN == 'EGNN':
            self.Encoder = layers.EGNN(nLigNode, nLigEdge, nHid, nLayer, dropout=0.1, JK=JK)
        elif GNN == 'AttentiveFP':
            self.Encoder = layers.ModifiedAttentiveFPGNNV2(nLigNode, nLigEdge, nLayer, nHid, 0.1, JK)

    def forward(self, Graph, Perturb=None):
        Node_Rep = self.Encoder(Graph, Perturb)
        return Node_Rep

class ASRP_head(nn.Module):
    def __init__(self, config):
        super(ASRP_head, self).__init__()

        self.readout = layers.ReadsOutLayer(config.model.inter_out_dim, config.model.readout, config.model.num_head, config.model.attn_merge)
        self.pretrain_assay_mlp_share = config.train.pretrain_assay_mlp_share

        if config.model.readout.startswith('multi_head') and config.model.attn_merge == 'concat':
            self.FC = layers.FC(config.model.inter_out_dim * (config.model.num_head + 1), config.model.fintune_fc_hidden_dim, config.model.dropout, config.model.out_dim)
        else:
            self.FC = layers.FC(config.model.inter_out_dim * 2, config.model.fintune_fc_hidden_dim, config.model.dropout, config.model.out_dim)

        self.regression_loss_fn = nn.MSELoss(reduce=False)
        self.ranking_loss_fn = losses.pairwise_BCE_loss(config)

    def forward(self, bg_inter, bond_feats_inter, ass_des, labels, select_flag):
        graph_embedding = self.readout(bg_inter, bond_feats_inter)

        affinity_pred = self.FC(graph_embedding)
        ranking_assay_embedding = torch.zeros(len(affinity_pred))

        y_pred_num = len(affinity_pred)
        assert y_pred_num % 2 == 0

        regression_loss = self.regression_loss_fn(affinity_pred[:y_pred_num // 2], labels[:y_pred_num // 2])  #
        labels_select = labels[:y_pred_num // 2][select_flag[:y_pred_num // 2]]
        affinity_pred_select = affinity_pred[:y_pred_num // 2][select_flag[:y_pred_num // 2]]
        regression_loss_select = regression_loss[select_flag[:y_pred_num // 2]].sum()

        ranking_loss, relation, relation_pred = self.ranking_loss_fn(graph_embedding, labels, ranking_assay_embedding)  #
        ranking_loss_select = ranking_loss[select_flag[:y_pred_num // 2]].sum()
        relation_select = relation[select_flag[:y_pred_num // 2]]
        relation_pred_selcet = relation_pred[select_flag[:y_pred_num // 2]]

        return regression_loss_select, ranking_loss_select,\
               labels_select, affinity_pred_select,\
               relation_select, relation_pred_selcet

    def inference(self, bg_inter, bond_feats_inter, ass_des, labels, select_flag):
        graph_embedding = self.readout(bg_inter, bond_feats_inter)
        affinity_pred = self.FC(graph_embedding)

        regression_loss = self.regression_loss_fn(affinity_pred, labels)  #
        regression_loss_select = regression_loss[select_flag].sum()

        labels_select = labels[select_flag]

        affinity_pred_select = affinity_pred[select_flag]

        return regression_loss_select, labels_select, affinity_pred_select

class Affinity_GNNs_MTL(nn.Module):
    def __init__(self, config):
        super(Affinity_GNNs_MTL, self).__init__()

        lig_node_dim = config.model.lig_node_dim
        lig_edge_dim = config.model.lig_edge_dim
        pro_node_dim = config.model.pro_node_dim
        pro_edge_dim = config.model.pro_edge_dim
        layer_num = config.model.num_layers
        hidden_dim = config.model.hidden_dim
        jk = config.model.jk
        GNN = config.model.GNN_type
        self.multi_task = config.train.multi_task

        self.pretrain_assay_mlp_share = config.train.pretrain_assay_mlp_share

        self.lig_encoder = GNNs(lig_node_dim, lig_edge_dim, layer_num, hidden_dim, jk, GNN)
        self.pro_encoder = GNNs(pro_node_dim, pro_edge_dim, layer_num, hidden_dim, jk, GNN)

        if config.model.jk == 'concat':
            self.noncov_graph = layers.DTIConvGraph3Layer(hidden_dim * (layer_num + layer_num) + config.model.inter_edge_dim, config.model.inter_out_dim, config.model.dropout)
        else:
            self.noncov_graph = layers.DTIConvGraph3Layer(hidden_dim * 2 + config.model.inter_edge_dim, config.model.inter_out_dim, config.model.dropout)
        self.softmax = nn.Softmax(dim=1)

        self.IC50_ASRP_head = ASRP_head(config)
        self.K_ASRP_head = ASRP_head(config)

        self.config = config

    def multi_head_inference(self, bg_inter, bond_feats_inter, labels, ass_des, IC50_f, K_f):
        regression_loss_IC50, affinity_IC50, affinity_pred_IC50 = \
            self.IC50_ASRP_head.inference(bg_inter, bond_feats_inter, ass_des, labels, IC50_f)

        regression_loss_K, affinity_K, affinity_pred_K = \
            self.K_ASRP_head.inference(bg_inter, bond_feats_inter, ass_des, labels, K_f)

        return (regression_loss_IC50, regression_loss_K),\
               (affinity_pred_IC50, affinity_pred_K), \
               (affinity_IC50, affinity_K)

    def multi_head_asrp(self, bg_inter, bond_feats_inter, labels, ass_des, IC50_f, K_f):
        regression_loss_IC50, ranking_loss_IC50, \
        affinity_IC50, affinity_pred_IC50, \
        relation_IC50, relation_pred_IC50 = self.IC50_ASRP_head(bg_inter, bond_feats_inter, ass_des, labels, IC50_f)

        regression_loss_K, ranking_loss_K, \
        affinity_K, affinity_pred_K, \
        relation_K, relation_pred_K = self.K_ASRP_head(bg_inter, bond_feats_inter, ass_des, labels, K_f)

        return (regression_loss_IC50, regression_loss_K),\
               (ranking_loss_IC50, ranking_loss_K), \
               (affinity_pred_IC50, affinity_pred_K), \
               (relation_pred_IC50, relation_pred_K), \
               (affinity_IC50, affinity_K), \
               (relation_IC50, relation_K)

    def alignfeature(self,bg_lig,bg_prot,node_feats_lig,node_feats_prot):
        inter_feature = torch.cat((node_feats_lig,node_feats_prot))
        lig_num,prot_num = bg_lig.batch_num_nodes(),bg_prot.batch_num_nodes()
        lig_start, prot_start = lig_num.cumsum(0) - lig_num, prot_num.cumsum(0) - prot_num
        inter_start = lig_start + prot_start
        for i in range(lig_num.shape[0]):
            inter_feature[inter_start[i]:inter_start[i]+lig_num[i]] = node_feats_lig[lig_start[i]:lig_start[i]+lig_num[i]]
            inter_feature[inter_start[i]+lig_num[i]:inter_start[i]+lig_num[i]+prot_num[i]] = node_feats_prot[prot_start[i]:prot_start[i]+prot_num[i]]
        return inter_feature


    def forward(self, batch, ASRP=True):
        bg_lig, bg_prot, bg_inter, labels, _, ass_des, IC50_f, K_f = batch
        node_feats_lig = self.lig_encoder(bg_lig)
        node_feats_prot = self.pro_encoder(bg_prot)
        bg_inter.ndata['h'] = self.alignfeature(bg_lig, bg_prot, node_feats_lig, node_feats_prot)
        bond_feats_inter = self.noncov_graph(bg_inter)

        if ASRP:
            return self.multi_head_asrp(bg_inter, bond_feats_inter, labels, ass_des, IC50_f, K_f)
        else:
            return self.multi_head_inference(bg_inter, bond_feats_inter, labels, ass_des, IC50_f, K_f)
