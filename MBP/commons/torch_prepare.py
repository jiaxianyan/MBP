import copy
import torch
import torch.nn as nn
import warnings
import dgl
import os
from MBP import runner, dataset
from .utils import get_run_dir

# customize exp lr scheduler with min lr
class ExponentialLR_with_minLr(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, min_lr=1e-4, last_epoch=-1, verbose=False):
        self.gamma = gamma
        self.min_lr = min_lr
        super(ExponentialLR_with_minLr, self).__init__(optimizer, gamma, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs
        return [max(group['lr'] * self.gamma, self.min_lr)
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
                for base_lr in self.base_lrs]


def get_scheduler(config, optimizer):
    if config.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.factor,
            patience=config.patience,
        )
    elif config.train.scheduler == 'expmin':
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=config.factor,
            min_lr=config.min_lr,
        )
    else:
        raise NotImplementedError('Scheduler not supported: %s' % config.type)

def get_optimizer(config, model):
    if config.type == "Adam":
        return torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=config.lr,
                        weight_decay=config.weight_decay)
    else:
        raise NotImplementedError('Optimizer not supported: %s' % config.type)

def get_optimizer_ablation(config, model, interact_ablation):
    if config.type == "Adam":
        return torch.optim.Adam(
                        filter(lambda p: p.requires_grad, list(model.parameters()) + list(interact_ablation.parameters()) ) ,
                        lr=config.lr,
                        weight_decay=config.weight_decay)
    else:
        raise NotImplementedError('Optimizer not supported: %s' % config.type)

def get_dataset(config, ddp=False):
    if config.data.dataset_name == 'chembl_in_pdbbind_smina':
        train_data, val_data = dataset.load_ChEMBL_Dock(config)

    return train_data, val_data, test_data

def get_finetune_dataset(config):
    train_data = dataset.pdbbind_finetune(config.data.finetune_train_names, config.data.finetune_dataset_name,
                                          config.data.labels_path, config)
    val_data = dataset.pdbbind_finetune(config.data.finetune_valid_names, config.data.finetune_dataset_name,
                                        config.data.labels_path, config)
    test_data = dataset.pdbbind_finetune(config.data.finetune_test_names, config.data.finetune_dataset_name,
                                         config.data.labels_path, config)

    generalize_csar_data = dataset.pdbbind_finetune(config.data.generalize_csar_test, config.data.generalize_dataset_name,
                                         config.data.generalize_labels_path, config)

    return train_data, val_data, test_data, generalize_csar_data

def get_test_dataset(config):
    test_data = dataset.pdbbind_finetune(config.data.finetune_test_names, config.data.finetune_dataset_name,
                                         config.data.labels_path, config)

    generalize_csar_data = dataset.pdbbind_finetune(config.data.generalize_csar_test, config.data.generalize_dataset_name,
                                         config.data.generalize_labels_path, config)

    return test_data, generalize_csar_data

def get_model(config):
    return globals()[config.model.model_type](config).to(config.train.device)

def repeat_data(data, num_repeat):
    datas = [copy.deepcopy(data) for i in range(num_repeat)]
    g_ligs, g_prots, g_inters = list(zip(*datas))
    return dgl.batch(g_ligs), dgl.batch(g_prots), dgl.batch(g_inters)

def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom