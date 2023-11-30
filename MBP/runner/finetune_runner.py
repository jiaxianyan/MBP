import dgl
import torch
import torch.nn as nn
from time import time
import os
from torch.utils.data import DataLoader
from MBP import dataset, commons, losses, models
import numpy as np
import pandas as pd

class DefaultRunner(object):
    def __init__(self,train_set, val_set, test_set, csar_set, model, optimizer, scheduler, config,
                 interact_ablation_model=None):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.csar_set = csar_set
        self.config = config

        self.device = config.train.device
        self.batch_size = self.config.train.batch_size
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler

        self.best_matric = 100
        self.start_epoch = 0

        if self.device.type == 'cuda':
            self._model = self._model.cuda(self.device)
        self.get_loss_fn()

        self.finetune_new_affinity_head = config.train.finetune_new_affinity_head
        if self.finetune_new_affinity_head:
            self.get_new_affinity_head()

        self.interact_ablation_model = interact_ablation_model

    def save(self, checkpoint, epoch=None, var_list={}):
        state = {
            **var_list,
            "model": self._model.state_dict(),
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
        # self.best_matric = state['best_matric']
        # self.start_epoch = state['cur_epoch'] + 1

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

    def get_new_affinity_head(self):
        self.affinity_new_head = models.affinity_head(self.config).to(self.device)

    def trans_device(self,batch):
        return [x if isinstance(x, list) else x.to(self.device) for x in batch]

    @torch.no_grad()
    def evaluate(self, split, verbose=0, logger=None, visualize=True):
        """
        Evaluate the model.
        Parameters:
            split (str): split to evaluate. Can be ``train``, ``val`` or ``test``.
        """

        test_set = getattr(self, "%s_set" % split)
        dataloader = DataLoader(test_set, batch_size=self.config.train.batch_size,
                                shuffle=False, collate_fn=dataset.collate_finetune,
                                num_workers=self.config.train.num_workers)

        y_preds, y_preds_IC50, y_preds_K = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        y, y_IC50, y_K = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        eval_start = time()
        model = self._model
        model.eval()
        for batch in dataloader:
            if self.device.type == "cuda":
                batch = self.trans_device(batch)

            (regression_loss_IC50, regression_loss_K), \
            (affinity_pred_IC50, affinity_pred_K), \
            (affinity_IC50, affinity_K) = model(batch, ASRP=False)

            affinity_pred = torch.cat([affinity_pred_IC50, affinity_pred_K], dim=0)
            affinity = torch.cat([affinity_IC50, affinity_K], dim=0)

            y_preds_IC50 = torch.cat([y_preds_IC50, affinity_pred_IC50])
            y_preds_K = torch.cat([y_preds_K, affinity_pred_K])
            y_preds = torch.cat([y_preds, affinity_pred])

            y_IC50 = torch.cat([y_IC50, affinity_IC50])
            y_K = torch.cat([y_K, affinity_K])
            y = torch.cat([y, affinity])

        metics_dict = commons.get_sbap_regression_metric_dict(np.array(y.cpu()), np.array(y_preds.cpu()))
        result_str = commons.get_matric_output_str(metics_dict)
        result_str = f'{split} total ' + result_str

        if len(y_IC50) > 0:
            metics_dict_IC50 = commons.get_sbap_regression_metric_dict(np.array(y_IC50.cpu()), np.array(y_preds_IC50.cpu()))
            result_str_IC50 = commons.get_matric_output_str(metics_dict_IC50)
            result_str_IC50 = f'| IC50 ' + result_str_IC50
            result_str += result_str_IC50

        if len(y_K) > 0:
            metics_dict_K = commons.get_sbap_regression_metric_dict(np.array(y_K.cpu()), np.array(y_preds_K.cpu()))
            result_str_K = commons.get_matric_output_str(metics_dict_K)
            result_str_K = f'| K ' + result_str_K
            result_str += result_str_K

        result_str += 'Time: %.4f'%(time() - eval_start)
        if verbose:
            if logger is not None:
                logger.info(result_str)
            else:
                print(result_str)
        return metics_dict['RMSE'], metics_dict['MAE'],  metics_dict['SD'], metics_dict['Pearson']

    def train(self, verbose=1, repeat_index=1):
        self.logger = self.config.logger
        self.logger.info(self.config)
        train_start = time()

        num_epochs = self.config.train.finetune_epochs

        dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size,
                                shuffle=self.config.train.shuffle, collate_fn=dataset.collate_finetune,
                                num_workers=self.config.train.num_workers, drop_last=True)

        model = self._model
        self.logger.info('trainable params in model: {:.2f}M'.format( sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))
        train_losses = []
        val_matric = []
        best_matric = self.best_matric
        start_epoch = self.start_epoch
        print('start training...')
        early_stop = 0

        for epoch in range(num_epochs):
            # train
            model.train()
            epoch_start = time()
            batch_losses, batch_ranking_losses = [], []
            batch_regression_ic50_losses, batch_regression_k_losses = [], []

            batch_cnt = 0
            for batch in dataloader:
                batch_cnt += 1
                if self.device.type == "cuda":
                    batch = self.trans_device(batch)

                (regression_loss_IC50, regression_loss_K), \
                (affinity_pred_IC50, affinity_pred_K), \
                (affinity_IC50, affinity_K) = model(batch, ASRP=False)

                regression_loss = regression_loss_IC50 + regression_loss_K

                if not regression_loss.requires_grad:
                    raise RuntimeError("loss doesn't require grad")

                self._optimizer.zero_grad()
                regression_loss.backward()
                self._optimizer.step()

                batch_losses.append(regression_loss.item())
                batch_regression_ic50_losses.append(regression_loss_IC50.item())
                batch_regression_k_losses.append(regression_loss_K.item())

            train_losses.append(sum(batch_losses))

            if verbose:
                self.logger.info('Epoch: %d | Train Loss: %.4f | '
                                 'Regression IC50 Loss: %.4f | Regression K Loss: %.4f | '
                                 'Lr: %.4f | Time: %.4f' % (
                epoch + start_epoch, sum(batch_losses),
                sum(batch_regression_ic50_losses), sum(batch_regression_k_losses),
                self._optimizer.param_groups[0]['lr'], time() - epoch_start))

            # evaluate
            if self.config.train.eval:
                eval_rmse = self.evaluate('val', verbose=1, logger=self.logger)
                val_matric.append(eval_rmse[0])

            if self.config.train.scheduler.type == "plateau":
                self._scheduler.step(train_losses[-1])
            else:
                self._scheduler.step()

            if val_matric[-1] < best_matric:
                early_stop = 0
                best_matric = val_matric[-1]
                if self.config.train.save:
                    print('saving checkpoint')
                    val_list = {
                        'cur_epoch': epoch + start_epoch,
                        'best_matric': best_matric,
                    }
                    self.save(self.config.train.save_path, f'best_valid_{repeat_index}', val_list)
                test_rmse, test_mae, tesr_sd, test_pearson = self.evaluate_mtl_v2('test', verbose=1, logger=self.logger)
            else:
                early_stop += 1
                if early_stop >= self.config.train.early_stop:
                    break

        self.best_matric = best_matric
        self.start_epoch = start_epoch + num_epochs
        print('optimization finished.')
        print('Total time elapsed: %.5fs' % (time() - train_start))

        return test_rmse, test_mae, tesr_sd, test_pearson