import torch
import torch.nn as nn
from time import time
import os
from torch.utils.data import DataLoader
from MBP import dataset, commons, losses
import numpy as np
import pandas as pd
import torch.distributed as dist

class DefaultRunner(object):
    def __init__(self,train_set, val_set, test_set, model, optimizer, scheduler, config):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.config = config

        self.device = config.train.device
        self.batch_size = self.config.train.batch_size
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler

        self.best_matric = -1


        self.start_epoch = 0

        if self.device.type == 'cuda':
            self._model = self._model.cuda(self.device)
        self.get_loss_fn()


    def save(self, checkpoint, epoch=None, ddp=False, var_list={}):
        state = {
            **var_list,
            "model": self._model.state_dict() if not ddp else self._model.module.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "config": self.config
        }
        epoch = str(epoch) if epoch is not None else ''
        checkpoint = os.path.join(checkpoint, 'checkpoint%s' % epoch)
        torch.save(state, checkpoint)

    def load(self, checkpoint, epoch=None, load_optimizer=False, load_scheduler=False):
        epoch = str(epoch) if epoch is not None else ''
        checkpoint = os.path.join(checkpoint, 'checkpoint%s' % epoch)
        print("Load checkpoint from %s" % checkpoint)

        state = torch.load(checkpoint, map_location=self.device)
        self._model.load_state_dict(state["model"])
        #self._model.load_state_dict(state["model"], strict=False)
        self.best_matric = state['best_matric']
        self.start_epoch = state['cur_epoch'] + 1

        if load_optimizer:
            self._optimizer.load_state_dict(state["optimizer"])
            if self.device.type == 'cuda':
                for state in self._optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.device)

        if load_scheduler:
            self._scheduler.load_state_dict(state["scheduler"])

    def get_loss_fn(self):
        self.loss_fn = nn.MSELoss()

    def trans_device(self,batch):
        return [x if isinstance(x, list) else x.to(self.device) for x in batch]

    def train(self, verbose=1, ddp=False):
        self.logger = self.config.logger
        train_start = time()

        num_epochs = self.config.train.pretrain_epochs

        if ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set)
            dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size, drop_last=True,
                                    collate_fn=dataset.collate_pretrain,
                                    num_workers=self.config.train.num_workers,
                                    sampler=train_sampler)
        else:
            dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size, drop_last=True,
                                    shuffle=self.config.train.shuffle, collate_fn=dataset.collate_pretrain,
                                    num_workers=self.config.train.num_workers)


        model = self._model
        if self.logger is not None:
            self.logger.info(self.config)
            self.logger.info('trainable params in model: {:.2f}M'.format( sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))
            self.logger.info('start training...')
        train_losses = []
        val_matric = []
        best_matric = self.best_matric
        best_loss = 1000000
        start_epoch = self.start_epoch

        early_stop = 0

        for epoch in range(num_epochs):
            # train
            model.train()
            epoch_start = time()
            batch_losses, batch_regression_losses, batch_ranking_losses = [], [], []
            batch_regression_ic50_losses, batch_regression_k_losses = [], []
            batch_ranking_ic50_losses, batch_ranking_k_losses = [], []
            batch_cnt = 0

            if ddp:
                dataloader.sampler.set_epoch(epoch)

            for batch in dataloader:
                batch_cnt += 1
                if self.device.type == "cuda":
                    batch = self.trans_device(batch)

                (regression_loss_IC50, regression_loss_K), \
                (ranking_loss_IC50, ranking_loss_K), \
                (affinity_pred_IC50, affinity_pred_K), \
                (relation_pred_IC50, relation_pred_K), \
                (affinity_IC50, affinity_K), \
                (relation_IC50, relation_K) = model(batch)

                regression_loss = self.config.train.pretrain_mtl_IC50_lambda * regression_loss_IC50 + \
                                  self.config.train.pretrain_mtl_K_lambda * regression_loss_K

                ranking_loss = self.config.train.pretrain_mtl_IC50_lambda * ranking_loss_IC50 + \
                               self.config.train.pretrain_mtl_K_lambda * ranking_loss_K

                pretrain_loss = self.config.train.pretrain_ranking_loss_lambda * ranking_loss +\
                                self.config.train.pretrain_regression_loss_lambda * regression_loss

                if not pretrain_loss.requires_grad:
                    raise RuntimeError("loss doesn't require grad")

                self._optimizer.zero_grad()
                pretrain_loss.backward()
                self._optimizer.step()

                batch_ranking_losses.append(self.config.train.pretrain_ranking_loss_lambda * ranking_loss.item())
                batch_regression_losses.append(self.config.train.pretrain_regression_loss_lambda * regression_loss.item())
                batch_losses.append(pretrain_loss.item())

                batch_regression_ic50_losses.append(self.config.train.pretrain_mtl_IC50_lambda * regression_loss_IC50.item())
                batch_regression_k_losses.append(self.config.train.pretrain_mtl_K_lambda * regression_loss_K.item())

                batch_ranking_ic50_losses.append(self.config.train.pretrain_mtl_IC50_lambda * ranking_loss_IC50.item())
                batch_ranking_k_losses.append(self.config.train.pretrain_mtl_K_lambda * ranking_loss_K.item())

            train_losses.append(sum(batch_losses))

            if self.logger is not None:
                self.logger.info('Epoch: %d | Pretrain Loss: %.4f | Regression Loss: %.4f | Ranking Loss: %.4f | '
                                 'Regression IC50 Loss: %.4f | Regression K Loss: %.4f | '
                                 'Ranking IC50 Loss: %.4f | Ranking K Loss: %.4f | Lr: %.4f | Time: %.4f' % (
                epoch + start_epoch, sum(batch_losses), sum(batch_regression_losses), sum(batch_ranking_losses),
                sum(batch_regression_ic50_losses), sum(batch_regression_k_losses),
                sum(batch_ranking_ic50_losses), sum(batch_ranking_k_losses),
                self._optimizer.param_groups[0]['lr'], time() - epoch_start))


            if (not ddp) or (ddp and dist.get_rank() == 0):
                if self.config.train.scheduler.type == "plateau":
                    self._scheduler.step(train_losses[-1])
                else:
                    self._scheduler.step()

                val_list = {
                    'cur_epoch': epoch + start_epoch,
                    'best_matric': best_matric,
                }

                self.save(self.config.train.save_path, 'latest', ddp, val_list)

            torch.cuda.empty_cache()
            if epoch % self.config.train.pretrain_regression_loss_lambda_degrade_epoch == 0:
                self.config.train.pretrain_regression_loss_lambda *= self.config.train.pretrain_regression_loss_lambda_degrade_ratio

        self.best_matric = best_matric
        self.start_epoch = start_epoch + num_epochs
        print('optimization finished.')
        print('Total time elapsed: %.5fs' % (time() - train_start))


