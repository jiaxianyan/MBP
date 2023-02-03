import torch
import torch.nn as nn
from time import time
import os
from torch.utils.data import DataLoader
from UltraFlow import dataset, commons, losses
import numpy as np
import pandas as pd

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

        if self.config.train.ranking_loss in ['pairwise_v1', 'pairwise_v2']:
            self.use_collate_fn = dataset.collate_affinity_pair_wise
        else:
            self.use_collate_fn = dataset.collate_affinity

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
        if self.config.train.ranking_loss == 'pairwise_v1':
            self.ranking_fn = losses.pair_wise_ranking_loss().to(self.device)
        elif self.config.train.ranking_loss == 'pairwise_v2':
            self.ranking_fn = losses.pair_wise_ranking_loss_v2(self.config).to(self.device)

    def trans_device(self,batch):
        return [x if isinstance(x, list) else x.to(self.device) for x in batch]

    @torch.no_grad()
    def evaluate(self, split, verbose=0, logger=None, visualize=True):
        """
        Evaluate the model.
        Parameters:
            split (str): split to evaluate. Can be ``train``, ``val`` or ``test``.
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError('split should be either train, val, or test.')

        test_set = getattr(self, "%s_set" % split)
        dataloader = DataLoader(test_set, batch_size=self.config.train.batch_size,
                                shuffle=False, collate_fn=dataset.collate_affinity,
                                num_workers=self.config.train.num_workers)
        y_preds = torch.tensor([]).to(self.device)
        y = torch.tensor([]).to(self.device)
        eval_start = time()
        model = self._model
        model.eval()
        for batch in dataloader:
            if self.device.type == "cuda":
                batch = self.trans_device(batch)
            y_pred, x_output = model(batch)
            y_preds = torch.cat([y_preds, y_pred])
            y = torch.cat([y, batch[-2]])
        np_y = np.array(y.cpu())
        np_f = np.array(y_preds.cpu())
        metics_dict = commons.get_sbap_matric_dict(np_y,np_f)
        result_str = commons.get_matric_output_str(metics_dict)
        result_str += 'Time: %.4f'%(time() - eval_start)
        if verbose:
            if logger is not None:
                logger.info(result_str)
            else:
                print(result_str)

        if visualize:
            result_d = {'pred_y': np_f.flatten().tolist(), 'y': np_y.flatten().tolist()}
            pd.DataFrame(result_d).to_csv(os.path.join(self.config.train.save_path, 'pred_values_pw_2.csv'))

        return metics_dict['Spearman']

    def train(self, verbose=1):
        self.logger = self.config.logger
        self.logger.info(self.config)
        train_start = time()

        num_epochs = self.config.train.epochs

        dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size,
                                shuffle=self.config.train.shuffle, collate_fn=self.use_collate_fn,
                                num_workers=self.config.train.num_workers)

        model = self._model
        self.logger.info('trainable params in model: {:.2f}M'.format( sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))
        train_losses = []
        val_matric = []
        best_matric = self.best_matric
        start_epoch = self.start_epoch
        print('start training...')
        early_stop = 0

        pair_wise_sample_num_total = [0] * len(self.train_set)
        for epoch in range(num_epochs):
            # train
            model.train()
            epoch_start = time()
            batch_losses, batch_ranking_losses = [], []
            batch_cnt = 0
            pair_wise_sample_num_epoch = [0] * len(self.train_set)
            for batch in dataloader:
                batch_cnt += 1
                if self.device.type == "cuda":
                    batch = self.trans_device(batch)
                y_pred, x_output = model(batch)
                loss = self.loss_fn(y_pred,batch[-2])

                if self.config.train.ranking_loss is not None:
                    if self.config.train.ranking_loss == 'pairwise_v1':
                        ranking_loss = self.ranking_fn(y_pred, batch[-2])
                    elif self.config.train.ranking_loss == 'pairwise_v2':
                        ranking_loss = self.ranking_fn(x_output, batch[-2])

                    batch_ranking_losses.append(ranking_loss.item())
                    loss += self.config.train.ranking_loss_lambda * ranking_loss

                if not loss.requires_grad:
                    raise RuntimeError("loss doesn't require grad")

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                batch_losses.append(loss.item())

                for index in batch[-1]:
                    pair_wise_sample_num_epoch[index] += 1
                    pair_wise_sample_num_total[index] += 1

            train_losses.append(sum(batch_losses))

            if verbose:
                self.logger.info('Epoch: %d | Train Loss: %.4f | Ranking Loss: %.4f | Lr: %.4f | Time: %.4f' % (
                epoch + start_epoch, sum(batch_losses), sum(batch_ranking_losses), self._optimizer.param_groups[0]['lr'], time() - epoch_start))

            # evaluate
            if self.config.train.eval:
                eval_rmse = self.evaluate('val', verbose=1, logger=self.logger)
                val_matric.append(eval_rmse)

            if self.config.train.scheduler.type == "plateau":
                self._scheduler.step(train_losses[-1])
            else:
                self._scheduler.step()

            if val_matric[-1] > best_matric:
                early_stop = 0
                best_matric = val_matric[-1]
                if self.config.train.save:
                    print('saving checkpoint')
                    val_list = {
                        'cur_epoch': epoch + start_epoch,
                        'best_matric': best_matric,
                    }
                    self.save(self.config.train.save_path, epoch + start_epoch, val_list)
                test_spearman = self.evaluate('test', verbose=1, logger=self.logger)
            else:
                early_stop += 1
                if early_stop >= self.config.train.early_stop:
                    break

            # record sample times
            if self.config.train.ranking_loss in ['pairwise_v1', 'pairwise_v2']:
                data_sample_epoch_d = {'data_index':list(range(len(self.train_set))),'sample_times':pair_wise_sample_num_epoch}
                data_sample_total_d = {'data_index':list(range(len(self.train_set))),'sample_times':pair_wise_sample_num_total}
                pd.DataFrame(data_sample_epoch_d).to_csv(os.path.join(self.config.train.save_path, f'epoch_{epoch + start_epoch}_sample_times.csv'))
                pd.DataFrame(data_sample_total_d).to_csv(os.path.join(self.config.train.save_path, f'total_sample_times.csv'))

        self.best_matric = best_matric
        self.start_epoch = start_epoch + num_epochs
        print('optimization finished.')
        print('Total time elapsed: %.5fs' % (time() - train_start))
