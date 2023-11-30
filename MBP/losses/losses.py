import numpy as np
import torch
import torch.nn as nn
from MBP import layers

# margin ranking loss
class pair_wise_ranking_loss(nn.Module):
    def __init__(self, config):
        super(pair_wise_ranking_loss, self).__init__()
        self.config = config
        self.threshold_filter = nn.Threshold(0.2, 0)
        self.score_predict = layers.FC(config.model.inter_out_dim * 2, config.model.fc_hidden_dim, config.model.dropout, 1)

    def ranking_loss(self, z_A, z_B, relation):
        """
        loss for a given set of pixels:
        z_A: predicted absolute depth for pixels A
        z_B: predicted absolute depth for pixels B
        relation: -1, 0, 1
        """
        pred_depth = z_A - z_B
        log_loss = torch.mean(torch.log(1 + torch.exp(-relation[relation != 0] * pred_depth[relation != 0])))
        return log_loss

    @torch.no_grad()
    def get_rank_relation(self, y_A, y_B):
        pred_depth = y_A - y_B
        pred_depth[self.threshold_filter(pred_depth.abs()) == 0] = 0

        return pred_depth.sign()

    def forward(self, output_embedding, target):
        batch_repeat_num = len(output_embedding)
        batch_size = batch_repeat_num // 2

        score_predict = self.score_predict(output_embedding)
        x_A, y_A, x_B, y_B = score_predict[:batch_size], target[:batch_size], score_predict[batch_size:], target[batch_size:]

        relation = self.get_rank_relation(y_A, y_B)

        ranking_loss = self.ranking_loss(x_A, x_B, relation)

        relation_pred = self.get_rank_relation(x_A, x_B)

        return ranking_loss, relation.squeeze(), relation_pred.squeeze()

# binary cross entropy loss
class pair_wise_ranking_loss_v2(nn.Module):
    def __init__(self, config):
        super(pair_wise_ranking_loss_v2, self).__init__()
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()
        self.relation_mlp = layers.FC(config.model.inter_out_dim * 4, [config.model.inter_out_dim * 2, config.model.inter_out_dim], config.model.dropout, 2)
        self.m = nn.Softmax(dim=1)

    @torch.no_grad()
    def get_rank_relation(self, y_A, y_B):
        # y_A: [batch, 1]
        # target_relation: 0: <=, 1: >
        target_relation = torch.zeros(y_A.size(), dtype=torch.long, device=y_A.device)
        target_relation[(y_A - y_B) > 0.0] = 1

        return target_relation.squeeze()

    def forward(self, output_embedding, target, assay_des):
        batch_repeat_num = len(output_embedding)
        batch_size = batch_repeat_num // 2
        x_A, y_A, x_B, y_B = output_embedding[:batch_size], target[:batch_size],\
                             output_embedding[batch_size:], target[batch_size:]

        relation = self.get_rank_relation(y_A, y_B)

        relation_pred = self.relation_mlp(torch.cat([x_A,x_B], dim=1))

        ranking_loss = self.loss_fn(relation_pred, relation)

        _, y_pred = self.m(relation_pred).max(dim=1)

        return ranking_loss, relation.squeeze(), y_pred

# binary cross entropy loss
class pairwise_BCE_loss(nn.Module):
    def __init__(self, config):
        super(pairwise_BCE_loss, self).__init__()
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss(reduce=False)
        if config.model.readout.startswith('multi_head') and config.model.attn_merge == 'concat':
            self.relation_mlp = layers.FC(config.model.inter_out_dim * (config.model.num_head + 1) * 2, [config.model.inter_out_dim * 2, config.model.inter_out_dim], config.model.dropout, 2)
        else:
            self.relation_mlp = layers.FC(config.model.inter_out_dim * 4, [config.model.inter_out_dim * 2, config.model.inter_out_dim], config.model.dropout, 2)
        self.m = nn.Softmax(dim=1)

    @torch.no_grad()
    def get_rank_relation(self, y_A, y_B):
        # y_A: [batch, 1]
        # target_relation: 0: <=, 1: >
        target_relation = torch.zeros(y_A.size(), dtype=torch.long, device=y_A.device)
        target_relation[(y_A - y_B) > 0.0] = 1

        return target_relation.squeeze()

    def forward(self, output_embedding, target, assay_des):
        batch_repeat_num = len(output_embedding)
        batch_size = batch_repeat_num // 2
        x_A, y_A, x_B, y_B = output_embedding[:batch_size], target[:batch_size],\
                             output_embedding[batch_size:], target[batch_size:]

        relation = self.get_rank_relation(y_A, y_B)

        relation_pred = self.relation_mlp(torch.cat([x_A,x_B], dim=1))

        ranking_loss = self.loss_fn(relation_pred, relation)

        _, y_pred = self.m(relation_pred).max(dim=1)

        return ranking_loss, relation.squeeze(), y_pred