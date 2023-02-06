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

    def forward(self, Graph):
        Node_Rep = self.Encoder(Graph)
        return Node_Rep
        
class Affinity_GNNs(nn.Module):
    def __init__(self, config):
        super(Affinity_GNNs, self).__init__()

        lig_node_dim = config.model.lig_node_dim
        lig_edge_dim = config.model.lig_edge_dim
        pro_node_dim = config.model.pro_node_dim
        pro_edge_dim = config.model.pro_edge_dim
        layer_num = config.model.num_layers
        hidden_dim = config.model.hidden_dim
        jk = config.model.jk
        GNN = config.model.GNN_type
        self.pretrain_assay_mlp_share = config.train.pretrain_assay_mlp_share
        self.pretrain_use_assay_description = config.train.pretrain_use_assay_description

        self.lig_encoder = GNNs(lig_node_dim, lig_edge_dim, layer_num, hidden_dim, jk, GNN)
        self.pro_encoder = GNNs(pro_node_dim, pro_edge_dim, layer_num, hidden_dim, jk, GNN)

        if config.model.jk == 'concat':
            self.noncov_graph = layers.DTIConvGraph3Layer(hidden_dim * (layer_num + layer_num) + config.model.inter_edge_dim, config.model.inter_out_dim, config.model.dropout)
        else:
            self.noncov_graph = layers.DTIConvGraph3Layer(hidden_dim * 2 + config.model.inter_edge_dim, config.model.inter_out_dim, config.model.dropout)
        self.readout = layers.ReadsOutLayer(config.model.inter_out_dim, config.model.readout)
        self.FC = layers.FC(config.model.inter_out_dim * 2, config.model.fc_hidden_dim, config.model.dropout, config.model.out_dim)
        self.softmax = nn.Softmax(dim=1)
        if self.pretrain_use_assay_description:
            print(f'use assay descrption type: {config.data.assay_des_type}')
            if self.pretrain_assay_mlp_share:
                self.assay_info_aggre_mlp = layers.FC(config.data.assay_des_dim, config.model.assay_des_fc_hidden_dim,
                                                      config.model.dropout, config.model.inter_out_dim * 2)
            else:
                self.assay_info_aggre_mlp_pointwise = layers.FC(config.data.assay_des_dim, config.model.assay_des_fc_hidden_dim,
                                                                config.model.dropout, config.model.inter_out_dim * 2)
                self.assay_info_aggre_mlp_pairwise = layers.FC(config.data.assay_des_dim, config.model.assay_des_fc_hidden_dim,
                                                               config.model.dropout, config.model.inter_out_dim * 2)

    def forward(self, batch):
        bg_lig, bg_prot, bg_inter, labels, _, ass_des = batch

        node_feats_lig = self.lig_encoder(bg_lig)
        node_feats_prot = self.pro_encoder(bg_prot)
        bg_inter.ndata['h'] = self.alignfeature(bg_lig,bg_prot,node_feats_lig,node_feats_prot)
        bond_feats_inter = self.noncov_graph(bg_inter)
        graph_embedding = self.readout(bg_inter, bond_feats_inter)

        if self.pretrain_use_assay_description:
            if self.pretrain_assay_mlp_share:
                ranking_assay_embedding = self.assay_info_aggre_mlp(ass_des)
                affinity_pred = self.FC(graph_embedding + ranking_assay_embedding)
            else:
                regression_assay_embedding = self.assay_info_aggre_mlp_pointwise(ass_des)
                affinity_pred = self.FC(graph_embedding + regression_assay_embedding)
                ranking_assay_embedding = self.assay_info_aggre_mlp_pairwise(ass_des)
        else:
            affinity_pred = self.FC(graph_embedding)
            ranking_assay_embedding = torch.zeros(len(affinity_pred))

        return affinity_pred, graph_embedding, ranking_assay_embedding

    def alignfeature(self,bg_lig,bg_prot,node_feats_lig,node_feats_prot):
        inter_feature = torch.cat((node_feats_lig,node_feats_prot))
        lig_num,prot_num = bg_lig.batch_num_nodes(),bg_prot.batch_num_nodes()
        lig_start, prot_start = lig_num.cumsum(0) - lig_num, prot_num.cumsum(0) - prot_num
        inter_start = lig_start + prot_start
        for i in range(lig_num.shape[0]):
            inter_feature[inter_start[i]:inter_start[i]+lig_num[i]] = node_feats_lig[lig_start[i]:lig_start[i]+lig_num[i]]
            inter_feature[inter_start[i]+lig_num[i]:inter_start[i]+lig_num[i]+prot_num[i]] = node_feats_prot[prot_start[i]:prot_start[i]+prot_num[i]]
        return inter_feature

class affinity_head(nn.Module):
    def __init__(self, config):
        super(affinity_head, self).__init__()
        self.pretrain_assay_mlp_share = config.train.pretrain_assay_mlp_share
        self.pretrain_use_assay_description = config.train.pretrain_use_assay_description
        if self.pretrain_use_assay_description:
            print(f'use assay descrption type: {config.data.assay_des_type}')
            if self.pretrain_assay_mlp_share:
                self.assay_info_aggre_mlp = layers.FC(config.data.assay_des_dim, config.model.assay_des_fc_hidden_dim,
                                                  config.model.dropout, config.model.inter_out_dim * 2)
            else:
                self.assay_info_aggre_mlp_pointwise = layers.FC(config.data.assay_des_dim, config.model.assay_des_fc_hidden_dim,
                                                        config.model.dropout, config.model.inter_out_dim * 2)
                self.assay_info_aggre_mlp_pairwise = layers.FC(config.data.assay_des_dim, config.model.assay_des_fc_hidden_dim,
                                                        config.model.dropout, config.model.inter_out_dim * 2)
        self.FC = layers.FC(config.model.inter_out_dim * 2, config.model.fintune_fc_hidden_dim, config.model.dropout, config.model.out_dim)

    def forward(self, graph_embedding, ass_des):
        if self.pretrain_use_assay_description:
            if self.pretrain_assay_mlp_share:
                ranking_assay_embedding = self.assay_info_aggre_mlp(ass_des)
                affinity_pred = self.FC(graph_embedding + ranking_assay_embedding)
            else:
                regression_assay_embedding = self.assay_info_aggre_mlp_pointwise(ass_des)
                affinity_pred = self.FC(graph_embedding + regression_assay_embedding)
                ranking_assay_embedding = self.assay_info_aggre_mlp_pairwise(ass_des)
        else:
            affinity_pred = self.FC(graph_embedding)
            ranking_assay_embedding = torch.zeros(len(affinity_pred))

        return affinity_pred


class ASRP_head(nn.Module):
    def __init__(self, config):
        super(ASRP_head, self).__init__()

        self.readout = layers.ReadsOutLayer(config.model.inter_out_dim, config.model.readout)
        self.pretrain_assay_mlp_share = config.train.pretrain_assay_mlp_share
        self.pretrain_use_assay_description = config.train.pretrain_use_assay_description
        if self.pretrain_use_assay_description:
            print(f'use assay descrption type: {config.data.assay_des_type}')
            if self.pretrain_assay_mlp_share:
                self.assay_info_aggre_mlp = layers.FC(config.data.assay_des_dim, config.model.assay_des_fc_hidden_dim,
                                                      config.model.dropout, config.model.inter_out_dim * 2)
            else:
                self.assay_info_aggre_mlp_pointwise = layers.FC(config.data.assay_des_dim, config.model.assay_des_fc_hidden_dim,
                                                                config.model.dropout, config.model.inter_out_dim * 2)
                self.assay_info_aggre_mlp_pairwise = layers.FC(config.data.assay_des_dim, config.model.assay_des_fc_hidden_dim,
                                                               config.model.dropout, config.model.inter_out_dim * 2)
        self.FC = layers.FC(config.model.inter_out_dim * 2, config.model.fintune_fc_hidden_dim, config.model.dropout,
                            config.model.out_dim)

        self.regression_loss_fn = nn.MSELoss(reduce=False)
        self.ranking_loss_fn = losses.pairwise_BCE_loss(config)

    def forward(self, bg_inter, bond_feats_inter, ass_des, labels, select_flag):
        graph_embedding = self.readout(bg_inter, bond_feats_inter)

        if self.pretrain_use_assay_description:
            if self.pretrain_assay_mlp_share:
                ranking_assay_embedding = self.assay_info_aggre_mlp(ass_des)
                affinity_pred = self.FC(graph_embedding + ranking_assay_embedding)
            else:
                regression_assay_embedding = self.assay_info_aggre_mlp_pointwise(ass_des)
                affinity_pred = self.FC(graph_embedding + regression_assay_embedding)
                ranking_assay_embedding = self.assay_info_aggre_mlp_pairwise(ass_des)
        else:
            affinity_pred = self.FC(graph_embedding)
            ranking_assay_embedding = torch.zeros(len(affinity_pred))

        y_pred_num = len(affinity_pred)
        assert y_pred_num % 2 == 0
        regression_loss = self.regression_loss_fn(affinity_pred[:y_pred_num // 2], labels[:y_pred_num // 2])  #
        ranking_loss, relation, relation_pred = self.ranking_loss_fn(graph_embedding, labels, ranking_assay_embedding)  #

        regression_loss_select = regression_loss[select_flag[:y_pred_num // 2]].sum()
        ranking_loss_select = ranking_loss[select_flag[:y_pred_num // 2]].sum()
        labels_select = labels[:y_pred_num // 2][select_flag[:y_pred_num // 2]]
        affinity_pred_select = affinity_pred[:y_pred_num // 2][select_flag[:y_pred_num // 2]]
        relation_select = relation[select_flag[:y_pred_num // 2]]
        relation_pred_selcet = relation_pred[select_flag[:y_pred_num // 2]]

        return regression_loss_select, ranking_loss_select,\
               labels_select, affinity_pred_select,\
               relation_select, relation_pred_selcet

    def forward_pointwise(self, bg_inter, bond_feats_inter, ass_des, labels, select_flag):
        graph_embedding = self.readout(bg_inter, bond_feats_inter)
        affinity_pred = self.FC(graph_embedding)

        regression_loss = self.regression_loss_fn(affinity_pred, labels)  #
        regression_loss_select = regression_loss[select_flag].sum()

        labels_select = labels[select_flag]

        affinity_pred_select = affinity_pred[select_flag]

        return regression_loss_select, labels_select, affinity_pred_select

    def evaluate_mtl(self, bg_inter, bond_feats_inter, ass_des, labels):
        graph_embedding = self.readout(bg_inter, bond_feats_inter)

        if self.pretrain_use_assay_description:
            if self.pretrain_assay_mlp_share:
                ranking_assay_embedding = self.assay_info_aggre_mlp(ass_des)
                affinity_pred = self.FC(graph_embedding + ranking_assay_embedding)
            else:
                regression_assay_embedding = self.assay_info_aggre_mlp_pointwise(ass_des)
                affinity_pred = self.FC(graph_embedding + regression_assay_embedding)
                ranking_assay_embedding = self.assay_info_aggre_mlp_pairwise(ass_des)
        else:
            affinity_pred = self.FC(graph_embedding)
            ranking_assay_embedding = torch.zeros(len(affinity_pred))

        n = graph_embedding.shape[0]
        pair_a_index, pair_b_index = [], []
        for i in range(n):
            pair_a_index.extend([i] * (n - 1))
            pair_b_index.extend([j for j in range(n) if i != j])

        pair_index = pair_a_index + pair_b_index

        _, relation, relation_pred = self.ranking_fn(graph_embedding[pair_index], labels[pair_index], ranking_assay_embedding[pair_index])

        return affinity_pred, relation, relation_pred


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
        self.pretrain_use_assay_description = config.train.pretrain_use_assay_description

        self.lig_encoder = GNNs(lig_node_dim, lig_edge_dim, layer_num, hidden_dim, jk, GNN)
        self.pro_encoder = GNNs(pro_node_dim, pro_edge_dim, layer_num, hidden_dim, jk, GNN)

        if config.model.jk == 'concat':
            self.noncov_graph = layers.DTIConvGraph3Layer(hidden_dim * (layer_num + layer_num) + config.model.inter_edge_dim, config.model.inter_out_dim, config.model.dropout)
        else:
            self.noncov_graph = layers.DTIConvGraph3Layer(hidden_dim * 2 + config.model.inter_edge_dim, config.model.inter_out_dim, config.model.dropout)
        self.softmax = nn.Softmax(dim=1)

        if self.multi_task == 'IC50KdKi':
            self.IC50_ASRP_head = ASRP_head(config)
            self.Kd_ASRP_head = ASRP_head(config)
            self.Ki_ASRP_head = ASRP_head(config)
        elif self.multi_task == 'IC50K':
            self.IC50_ASRP_head = ASRP_head(config)
            self.K_ASRP_head = ASRP_head(config)

    def forward(self, batch, ASRP=True):
        if self.multi_task == 'IC50KdKi':
            bg_lig, bg_prot, bg_inter, labels, _, ass_des, IC50_f, Kd_f, Ki_f = batch
            node_feats_lig = self.lig_encoder(bg_lig)
            node_feats_prot = self.pro_encoder(bg_prot)
            bg_inter.ndata['h'] = self.alignfeature(bg_lig,bg_prot,node_feats_lig,node_feats_prot)
            bond_feats_inter = self.noncov_graph(bg_inter)
            if ASRP:
                return self.multi_head_pred(bg_inter, bond_feats_inter, labels, ass_des, IC50_f, Kd_f, Ki_f)
            else:
                return self.multi_head_pointwise(bg_inter, bond_feats_inter, labels, ass_des, IC50_f, Kd_f, Ki_f)

        elif self.multi_task == 'IC50K':
            bg_lig, bg_prot, bg_inter, labels, _, ass_des, IC50_f, K_f = batch
            node_feats_lig = self.lig_encoder(bg_lig)
            node_feats_prot = self.pro_encoder(bg_prot)
            bg_inter.ndata['h'] = self.alignfeature(bg_lig,bg_prot,node_feats_lig,node_feats_prot)
            bond_feats_inter = self.noncov_graph(bg_inter)
            if ASRP:
                return self.multi_head_pred_v2(bg_inter, bond_feats_inter, labels, ass_des, IC50_f, K_f)
            else:
                return self.multi_head_pointwise_v2(bg_inter, bond_feats_inter, labels, ass_des, IC50_f, K_f)

    def multi_head_pointwise(self, bg_inter, bond_feats_inter, labels, ass_des, IC50_f, Kd_f, Ki_f):
        regression_loss_IC50, affinity_IC50, affinity_pred_IC50 = \
            self.IC50_ASRP_head.forward_pointwise(bg_inter, bond_feats_inter, ass_des, labels, IC50_f)

        regression_loss_Kd, affinity_Kd, affinity_pred_Kd = \
            self.Kd_ASRP_head.forward_pointwise(bg_inter, bond_feats_inter, ass_des, labels, Kd_f)

        regression_loss_Ki, affinity_Ki, affinity_pred_Ki = \
            self.Ki_ASRP_head.forward_pointwise(bg_inter, bond_feats_inter, ass_des, labels, Ki_f)

        return (regression_loss_IC50, regression_loss_Kd, regression_loss_Ki),\
               (affinity_pred_IC50, affinity_pred_Kd, affinity_pred_Ki), \
               (affinity_IC50, affinity_Kd, affinity_Ki)

    def multi_head_pointwise_v2(self, bg_inter, bond_feats_inter, labels, ass_des, IC50_f, K_f):
        regression_loss_IC50, affinity_IC50, affinity_pred_IC50 = \
            self.IC50_ASRP_head.forward_pointwise(bg_inter, bond_feats_inter, ass_des, labels, IC50_f)

        regression_loss_K, affinity_K, affinity_pred_K = \
            self.K_ASRP_head.forward_pointwise(bg_inter, bond_feats_inter, ass_des, labels, K_f)

        return (regression_loss_IC50, regression_loss_K),\
               (affinity_pred_IC50, affinity_pred_K), \
               (affinity_IC50, affinity_K)


    def multi_head_pred(self, bg_inter, bond_feats_inter, labels, ass_des, IC50_f, Kd_f, Ki_f):
        regression_loss_IC50, ranking_loss_IC50, \
        affinity_IC50, affinity_pred_IC50, \
        relation_IC50, relation_pred_IC50 = self.IC50_ASRP_head(bg_inter, bond_feats_inter, ass_des, labels, IC50_f)

        regression_loss_Kd, ranking_loss_Kd, \
        affinity_Kd, affinity_pred_Kd, \
        relation_Kd, relation_pred_Kd = self.Kd_ASRP_head(bg_inter, bond_feats_inter, ass_des, labels, Kd_f)

        regression_loss_Ki, ranking_loss_Ki, \
        affinity_Ki, affinity_pred_Ki, \
        relation_Ki, relation_pred_Ki = self.Ki_ASRP_head(bg_inter, bond_feats_inter, ass_des, labels, Ki_f)


        return (regression_loss_IC50, regression_loss_Kd, regression_loss_Ki),\
               (ranking_loss_IC50, ranking_loss_Kd, ranking_loss_Ki), \
               (affinity_pred_IC50, affinity_pred_Kd, affinity_pred_Ki), \
               (relation_pred_IC50, relation_pred_Kd, relation_pred_Ki), \
               (affinity_IC50, affinity_Kd, affinity_Ki), \
               (relation_IC50, relation_Kd, relation_Kd)

    def multi_head_pred_v2(self, bg_inter, bond_feats_inter, labels, ass_des, IC50_f, K_f):
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

    def multi_head_evaluate(self, bg_inter, bond_feats_inter, labels, ass_des, IC50_f, Kd_f, Ki_f):
        if sum(IC50_f):
            assert sum(Kd_f) == 0 and sum(Ki_f) == 0
            return self.IC50_ASRP_head.evaluate_mtl(bg_inter, bond_feats_inter, labels, ass_des)
        elif sum(Kd_f):
            assert sum(IC50_f) == 0 and sum(Ki_f) == 0
            return self.Kd_ASRP_head.evaluate_mtl(bg_inter, bond_feats_inter, labels, ass_des)
        elif sum(Ki_f):
            assert sum(IC50_f) == 0 and sum(Kd_f) == 0
            return self.Kd_ASRP_head.evaluate_mtl(bg_inter, bond_feats_inter, labels, ass_des)

    def alignfeature(self,bg_lig,bg_prot,node_feats_lig,node_feats_prot):
        inter_feature = torch.cat((node_feats_lig,node_feats_prot))
        lig_num,prot_num = bg_lig.batch_num_nodes(),bg_prot.batch_num_nodes()
        lig_start, prot_start = lig_num.cumsum(0) - lig_num, prot_num.cumsum(0) - prot_num
        inter_start = lig_start + prot_start
        for i in range(lig_num.shape[0]):
            inter_feature[inter_start[i]:inter_start[i]+lig_num[i]] = node_feats_lig[lig_start[i]:lig_start[i]+lig_num[i]]
            inter_feature[inter_start[i]+lig_num[i]:inter_start[i]+lig_num[i]+prot_num[i]] = node_feats_prot[prot_start[i]:prot_start[i]+prot_num[i]]
        return inter_feature


class IGN_basic(nn.Module):
    def __init__(self,config):
        super(IGN_basic, self).__init__()
        self.config = config
        self.pretrain_assay_mlp_share = config.train.pretrain_assay_mlp_share
        self.pretrain_use_assay_description = config.train.pretrain_use_assay_description
        self.graph_conv = layers.ModifiedAttentiveFPGNNV2(config.model.lig_node_dim, config.model.lig_edge_dim, config.model.num_layers, config.model.hidden_dim, config.model.dropout, config.model.jk)
        if config.model.jk == 'concat':
            self.noncov_graph = layers.DTIConvGraph3Layer_IGN_basic(config.model.hidden_dim * config.model.num_layers + config.model.inter_edge_dim, config.model.inter_out_dim, config.model.dropout)
        else:
            self.noncov_graph = layers.DTIConvGraph3Layer_IGN_basic(config.model.hidden_dim + config.model.inter_edge_dim, config.model.inter_out_dim, config.model.dropout)

        self.FC = layers.FC(config.model.inter_out_dim * 2, config.model.fc_hidden_dim, config.model.dropout, config.model.out_dim)
        self.readout = layers.ReadsOutLayer(config.model.inter_out_dim, config.model.readout)
        self.softmax = nn.Softmax(dim=1)
        if self.pretrain_use_assay_description:
            print(f'use assay descrption type: {config.data.assay_des_type}')
            if self.pretrain_assay_mlp_share:
                self.assay_info_aggre_mlp = layers.FC(config.data.assay_des_dim, config.model.assay_des_fc_hidden_dim,
                                                  config.model.dropout, config.model.inter_out_dim * 2)
            else:
                self.assay_info_aggre_mlp_pointwise = layers.FC(config.data.assay_des_dim, config.model.assay_des_fc_hidden_dim,
                                                        config.model.dropout, config.model.inter_out_dim * 2)
                self.assay_info_aggre_mlp_pairwise = layers.FC(config.data.assay_des_dim, config.model.assay_des_fc_hidden_dim,
                                                        config.model.dropout, config.model.inter_out_dim * 2)

    def forward(self, batch):
        bg_lig, bg_prot, bg_inter, labels, _, ass_des = batch

        node_feats_lig = self.graph_conv(bg_lig)
        node_feats_prot = self.graph_conv(bg_prot)
        bg_inter.ndata['h'] = self.alignfeature(bg_lig,bg_prot,node_feats_lig,node_feats_prot)
        bond_feats_inter = self.noncov_graph(bg_inter)
        graph_embedding = self.readout(bg_inter, bond_feats_inter)

        if self.pretrain_use_assay_description:
            if self.pretrain_assay_mlp_share:
                ranking_assay_embedding = self.assay_info_aggre_mlp(ass_des)
                affinity_pred = self.FC(graph_embedding + ranking_assay_embedding)
            else:
                regression_assay_embedding = self.assay_info_aggre_mlp_pointwise(ass_des)
                affinity_pred = self.FC(graph_embedding + regression_assay_embedding)
                ranking_assay_embedding = self.assay_info_aggre_mlp_pairwise(ass_des)
        else:
            affinity_pred = self.FC(graph_embedding)
            ranking_assay_embedding = torch.zeros(len(affinity_pred))

        return affinity_pred, graph_embedding, ranking_assay_embedding

    def alignfeature(self,bg_lig,bg_prot,node_feats_lig,node_feats_prot):
        inter_feature = torch.cat((node_feats_lig,node_feats_prot))
        lig_num,prot_num = bg_lig.batch_num_nodes(),bg_prot.batch_num_nodes()
        lig_start, prot_start = lig_num.cumsum(0) - lig_num, prot_num.cumsum(0) - prot_num
        inter_start = lig_start + prot_start
        for i in range(lig_num.shape[0]):
            inter_feature[inter_start[i]:inter_start[i]+lig_num[i]] = node_feats_lig[lig_start[i]:lig_start[i]+lig_num[i]]
            inter_feature[inter_start[i]+lig_num[i]:inter_start[i]+lig_num[i]+prot_num[i]] = node_feats_prot[prot_start[i]:prot_start[i]+prot_num[i]]
        return inter_feature

class IGN(nn.Module):
    def __init__(self,config):
        super(IGN, self).__init__()
        self.config = config
        self.pretrain_assay_mlp_share = config.train.pretrain_assay_mlp_share
        self.pretrain_use_assay_description = config.train.pretrain_use_assay_description
        self.ligand_conv = layers.ModifiedAttentiveFPGNNV2(config.model.lig_node_dim, config.model.lig_edge_dim, config.model.num_layers, config.model.hidden_dim, config.model.dropout, config.model.jk)
        self.protein_conv = layers.ModifiedAttentiveFPGNNV2(config.model.pro_node_dim, config.model.pro_edge_dim, config.model.num_layers, config.model.hidden_dim, config.model.dropout, config.model.jk)
        if config.model.jk == 'concat':
            self.noncov_graph = layers.DTIConvGraph3Layer(config.model.hidden_dim * (config.model.num_layers + config.model.num_layers) + config.model.inter_edge_dim, config.model.inter_out_dim, config.model.dropout)
        else:
            self.noncov_graph = layers.DTIConvGraph3Layer(config.model.hidden_dim * 2 + config.model.inter_edge_dim, config.model.inter_out_dim, config.model.dropout)

        self.FC = layers.FC(config.model.inter_out_dim * 2, config.model.fc_hidden_dim, config.model.dropout, config.model.out_dim)
        self.readout = layers.ReadsOutLayer(config.model.inter_out_dim, config.model.readout)
        self.softmax = nn.Softmax(dim=1)
        if self.pretrain_use_assay_description:
            print(f'use assay descrption type: {config.data.assay_des_type}')
            if self.pretrain_assay_mlp_share:
                self.assay_info_aggre_mlp = layers.FC(config.data.assay_des_dim, config.model.assay_des_fc_hidden_dim,
                                                  config.model.dropout, config.model.inter_out_dim * 2)
            else:
                self.assay_info_aggre_mlp_pointwise = layers.FC(config.data.assay_des_dim, config.model.assay_des_fc_hidden_dim,
                                                        config.model.dropout, config.model.inter_out_dim * 2)
                self.assay_info_aggre_mlp_pairwise = layers.FC(config.data.assay_des_dim, config.model.assay_des_fc_hidden_dim,
                                                        config.model.dropout, config.model.inter_out_dim * 2)

    def forward(self, batch):
        bg_lig, bg_prot, bg_inter, labels, _, ass_des = batch

        node_feats_lig = self.ligand_conv(bg_lig)
        node_feats_prot = self.protein_conv(bg_prot)
        bg_inter.ndata['h'] = self.alignfeature(bg_lig,bg_prot,node_feats_lig,node_feats_prot)
        bond_feats_inter = self.noncov_graph(bg_inter)
        graph_embedding = self.readout(bg_inter, bond_feats_inter)

        if self.pretrain_use_assay_description:
            if self.pretrain_assay_mlp_share:
                ranking_assay_embedding = self.assay_info_aggre_mlp(ass_des)
                affinity_pred = self.FC(graph_embedding + ranking_assay_embedding)
            else:
                regression_assay_embedding = self.assay_info_aggre_mlp_pointwise(ass_des)
                affinity_pred = self.FC(graph_embedding + regression_assay_embedding)
                ranking_assay_embedding = self.assay_info_aggre_mlp_pairwise(ass_des)
        else:
            affinity_pred = self.FC(graph_embedding)
            ranking_assay_embedding = torch.zeros(len(affinity_pred))

        return affinity_pred, graph_embedding, ranking_assay_embedding

    def alignfeature(self,bg_lig,bg_prot,node_feats_lig,node_feats_prot):
        inter_feature = torch.cat((node_feats_lig,node_feats_prot))
        lig_num,prot_num = bg_lig.batch_num_nodes(),bg_prot.batch_num_nodes()
        lig_start, prot_start = lig_num.cumsum(0) - lig_num, prot_num.cumsum(0) - prot_num
        inter_start = lig_start + prot_start
        for i in range(lig_num.shape[0]):
            inter_feature[inter_start[i]:inter_start[i]+lig_num[i]] = node_feats_lig[lig_start[i]:lig_start[i]+lig_num[i]]
            inter_feature[inter_start[i]+lig_num[i]:inter_start[i]+lig_num[i]+prot_num[i]] = node_feats_prot[prot_start[i]:prot_start[i]+prot_num[i]]
        return inter_feature
