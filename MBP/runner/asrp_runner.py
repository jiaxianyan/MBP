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
        if self.config.train.pretrain_ranking_loss == 'pairwise_v2':
            self.ranking_fn = losses.pair_wise_ranking_loss_v2(self.config).to(self.device)

    def trans_device(self,batch):
        return [x if isinstance(x, list) else x.to(self.device) for x in batch]

    @torch.no_grad()
    def evaluate_pairwsie(self, split, verbose=0, logger=None, visualize=True):
        """
        Evaluate the model.
        Parameters:
            split (str): split to evaluate. Can be ``train``, ``val`` or ``test``.
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError('split should be either train, val, or test.')

        test_set = getattr(self, "%s_set" % split)

        relation_preds = torch.tensor([]).to(self.device)
        relations = torch.tensor([]).to(self.device)
        y_preds = torch.tensor([]).to(self.device)
        ys = torch.tensor([]).to(self.device)
        eval_start = time()
        model = self._model
        model.eval()
        for batch in test_set:
            if self.device.type == "cuda":
                batch = self.trans_device(batch)

            y_pred, x_output, ranking_assay_embedding = model(batch)

            n = x_output.shape[0]
            pair_a_index, pair_b_index = [], []
            for i in range(n):
                pair_a_index.extend([i] * (n - 1))
                pair_b_index.extend([j for j in range(n) if i != j])

            pair_index = pair_a_index + pair_b_index

            _, relation, relation_pred = self.ranking_fn(x_output[pair_index], batch[-3][pair_index], ranking_assay_embedding[pair_index])

            relation_preds = torch.cat([relation_preds, relation_pred])
            relations = torch.cat([relations, relation])

            y_preds = torch.cat([y_preds, y_pred])
            ys = torch.cat([ys, batch[-3]])

        acc = (sum(relation_preds == relations) / (len(relation_preds))).cpu().item()
        result_str = 'valid acc: {:.4f}'.format(acc)

        np_y = np.array(ys.cpu())
        np_f = np.array(y_preds.cpu())
        regression_metrics_dict = commons.get_sbap_regression_metric_dict(np_y, np_f)
        regression_result_str = commons.get_matric_output_str(regression_metrics_dict)
        result_str += regression_result_str

        result_str += ' | Time: %.4f'%(time() - eval_start)

        if verbose:
            if logger is not None:
                logger.info(result_str)
            else:
                print(result_str)

        return acc

    @torch.no_grad()
    def evaluate_pointwise(self, split, verbose=0, logger=None, visualize=True):
        """
        Evaluate the model.
        Parameters:
            split (str): split to evaluate. Can be ``train``, ``val`` or ``test``.
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError('split should be either train, val, or test.')

        test_set = getattr(self, "%s_set" % split)

        relation_preds = torch.tensor([]).to(self.device)
        relations = torch.tensor([]).to(self.device)
        y_preds = torch.tensor([]).to(self.device)
        ys = torch.tensor([]).to(self.device)
        eval_start = time()
        model = self._model
        model.eval()
        for batch in test_set:
            if self.device.type == "cuda":
                batch = self.trans_device(batch)

            y_pred, x_output, _ = model(batch)

            n = x_output.shape[0]
            pair_a_index, pair_b_index = [], []
            for i in range(n):
                pair_a_index.extend([i] * (n - 1))
                pair_b_index.extend([j for j in range(n) if i != j])

            pair_index = pair_a_index + pair_b_index

            score_pred = y_pred[pair_index]
            score_target = batch[-3][pair_index]

            batch_repeat_num = len(score_pred)
            batch_size = batch_repeat_num // 2
            pred_A, targe_A, pred_B, target_B = score_pred[:batch_size], score_target[:batch_size], score_pred[batch_size:], score_target[batch_size:]

            relation_pred = torch.zeros(pred_A.size(), dtype=torch.long, device=pred_A.device)
            relation_pred[(pred_A - pred_B) > 0.0] = 1

            relation = torch.zeros(targe_A.size(), dtype=torch.long, device=targe_A.device)
            relation[(targe_A - target_B) > 0.0] = 1

            relation_preds = torch.cat([relation_preds, relation_pred])
            relations = torch.cat([relations, relation])

            y_preds = torch.cat([y_preds, y_pred])
            ys = torch.cat([ys, batch[-3]])

        acc = (sum(relation_preds == relations) / (len(relation_preds))).cpu().item()
        result_str = 'valid acc: {:.4f}'.format(acc)

        np_y = np.array(ys.cpu())
        np_f = np.array(y_preds.cpu())
        regression_metrics_dict = commons.get_sbap_regression_metric_dict(np_y, np_f)
        regression_result_str = commons.get_matric_output_str(regression_metrics_dict)
        result_str += regression_result_str

        result_str += ' | Time: %.4f'%(time() - eval_start)

        if verbose:
            if logger is not None:
                logger.info(result_str)
            else:
                print(result_str)

        return acc, regression_metrics_dict['RMSE']

    def train(self, ddp=False):
        if self.config.train.pretrain_sampling_method == 'pairwise_v1':
            if not self.config.train.multi_task:
                print('begin pairwise_v1 training')
                self.train_pairwise_v1(ddp=ddp)
            elif self.config.train.multi_task == 'IC50KdKi':
                print('begin pairwise_v1 multi-task training IC50/Kd/Ki')
                self.train_pairwise_v1_multi_task(ddp=ddp)
            elif self.config.train.multi_task == 'IC50K':
                print('begin pairwise_v1 multi-task training IC50/K')
                self.train_pairwise_v1_multi_task_v2(ddp=ddp)

        elif self.config.train.pretrain_sampling_method == 'pointwise':
            if not self.config.train.multi_task:
                print('begin pointwise training')
                self.train_pointwise(ddp=ddp)
            elif self.config.train.multi_task == 'IC50KdKi':
                print('begin pointwise multi-task training IC50/Kd/Ki')
                self.train_pointwise_multi_task(ddp=ddp)
            elif self.config.train.multi_task == 'IC50K':
                print('begin pointwise multi-task training IC50/K')
                self.train_pointwise_multi_task_v2(ddp=ddp)

    def train_pairwise_v1(self, verbose=1, ddp=False):
        self.logger = self.config.logger
        train_start = time()

        num_epochs = self.config.train.pretrain_epochs

        if ddp and self.config.train.use_memory_efficient_dataset != 'v1':
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set)
            dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size, drop_last=True,
                                    collate_fn=dataset.collate_affinity_pair_wise,
                                    num_workers=self.config.train.num_workers,
                                    sampler=train_sampler)
        else:
            dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size, drop_last=True,
                                    shuffle=self.config.train.shuffle, collate_fn=dataset.collate_affinity_pair_wise,
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
            batch_cnt = 0

            if ddp and self.config.train.use_memory_efficient_dataset != 'v1':
                dataloader.sampler.set_epoch(epoch)

            for batch in dataloader:
                batch_cnt += 1
                if self.device.type == "cuda":
                    batch = self.trans_device(batch)

                y_pred, x_output, ranking_assay_embedding = model(batch)

                y_pred_num = len(y_pred)
                assert y_pred_num % 2 == 0

                if self.config.train.pairwise_two_tower_regression_loss:
                    regression_loss = self.loss_fn(y_pred, batch[-3])
                else:
                    regression_loss = self.loss_fn(y_pred[:y_pred_num // 2], batch[-3][:y_pred_num // 2])

                ranking_loss, _, _ = self.ranking_fn(x_output, batch[-3], ranking_assay_embedding)

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

            train_losses.append(sum(batch_losses))

            if self.logger is not None:
                self.logger.info('Epoch: %d | Pretrain Loss: %.4f | Regression Loss: %.4f | Ranking Loss: %.4f | Lr: %.4f | Time: %.4f' % (
                epoch + start_epoch, sum(batch_losses), sum(batch_regression_losses), sum(batch_ranking_losses), self._optimizer.param_groups[0]['lr'], time() - epoch_start))


            if (not ddp) or (ddp and dist.get_rank() == 0):
                # evaluate
                if self.config.train.eval:
                    eval_acc = self.evaluate_pairwsie('val', verbose=1, logger=self.logger)
                    val_matric.append(eval_acc)

                if self.config.train.scheduler.type == "plateau":
                    self._scheduler.step(train_losses[-1])
                else:
                    self._scheduler.step()

                val_list = {
                    'cur_epoch': epoch + start_epoch,
                    'best_matric': best_matric,
                }

                self.save(self.config.train.save_path, 'latest', ddp, val_list)

                if sum(batch_losses) < best_loss:
                    best_loss = sum(batch_losses)
                    self.save(self.config.train.save_path, 'best_loss', ddp, val_list)

                if val_matric[-1] > best_matric:
                    early_stop = 0
                    best_matric = val_matric[-1]
                    if self.config.train.save:
                        print('saving checkpoint')
                        val_list = {
                            'cur_epoch': epoch + start_epoch,
                            'best_matric': best_matric,
                        }
                        self.save(self.config.train.save_path, epoch + start_epoch, ddp, val_list)

            torch.cuda.empty_cache()
            if epoch % self.config.train.pretrain_regression_loss_lambda_degrade_epoch == 0:
                self.config.train.pretrain_regression_loss_lambda *= self.config.train.pretrain_regression_loss_lambda_degrade_ratio

        self.best_matric = best_matric
        self.start_epoch = start_epoch + num_epochs
        print('optimization finished.')
        print('Total time elapsed: %.5fs' % (time() - train_start))

    def train_pairwise_v1_multi_task(self, verbose=1, ddp=False):
        self.logger = self.config.logger
        train_start = time()

        num_epochs = self.config.train.pretrain_epochs

        if ddp and self.config.train.use_memory_efficient_dataset != 'v1':
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set)
            dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size, drop_last=True,
                                    collate_fn=dataset.collate_affinity_pair_wise_multi_task,
                                    num_workers=self.config.train.num_workers,
                                    sampler=train_sampler)
        else:
            dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size, drop_last=True,
                                    shuffle=self.config.train.shuffle, collate_fn=dataset.collate_affinity_pair_wise_multi_task,
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
            batch_regression_ic50_losses, batch_regression_kd_losses, batch_regression_ki_losses = [], [], []
            batch_ranking_ic50_losses, batch_ranking_kd_losses, batch_ranking_ki_losses = [], [], []
            batch_cnt = 0

            if ddp and self.config.train.use_memory_efficient_dataset != 'v1':
                dataloader.sampler.set_epoch(epoch)

            for batch in dataloader:
                batch_cnt += 1
                if self.device.type == "cuda":
                    batch = self.trans_device(batch)

                (regression_loss_IC50, regression_loss_Kd, regression_loss_Ki), \
                (ranking_loss_IC50, ranking_loss_Kd, ranking_loss_Ki), \
                (affinity_pred_IC50, affinity_pred_Kd, affinity_pred_Ki), \
                (relation_pred_IC50, relation_pred_Kd, relation_pred_Ki), \
                (affinity_IC50, affinity_Kd, affinity_Ki), \
                (relation_IC50, relation_Kd, relation_Kd) = model(batch)

                regression_loss = self.config.train.pretrain_mtl_IC50_lambda * regression_loss_IC50 + \
                                  self.config.train.pretrain_mtl_Kd_lambda * regression_loss_Kd + \
                                  self.config.train.pretrain_mtl_Ki_lambda * regression_loss_Ki

                ranking_loss = self.config.train.pretrain_mtl_IC50_lambda * ranking_loss_IC50 + \
                               self.config.train.pretrain_mtl_Kd_lambda * ranking_loss_Kd + \
                               self.config.train.pretrain_mtl_Kd_lambda * ranking_loss_Ki


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
                batch_regression_kd_losses.append(self.config.train.pretrain_mtl_Kd_lambda * regression_loss_Kd.item())
                batch_regression_ki_losses.append(self.config.train.pretrain_mtl_Ki_lambda * regression_loss_Ki.item())

                batch_ranking_ic50_losses.append(self.config.train.pretrain_mtl_IC50_lambda * ranking_loss_IC50.item())
                batch_ranking_kd_losses.append(self.config.train.pretrain_mtl_Kd_lambda * ranking_loss_Kd.item())
                batch_ranking_ki_losses.append(self.config.train.pretrain_mtl_Ki_lambda * ranking_loss_Ki.item())

            train_losses.append(sum(batch_losses))

            if self.logger is not None:
                self.logger.info('Epoch: %d | Pretrain Loss: %.4f | Regression Loss: %.4f | Ranking Loss: %.4f | '
                                 'Regression IC50 Loss: %.4f | Regression Kd Loss: %.4f | Regression Ki Loss: %.4f | '
                                 'Ranking IC50 Loss: %.4f | Ranking Kd Loss: %.4f | Ranking Ki Loss: %.4f | Lr: %.4f | Time: %.4f' % (
                epoch + start_epoch, sum(batch_losses), sum(batch_regression_losses), sum(batch_ranking_losses),
                sum(batch_regression_ic50_losses), sum(batch_regression_kd_losses), sum(batch_regression_ki_losses),
                sum(batch_ranking_ic50_losses), sum(batch_ranking_kd_losses), sum(batch_ranking_ki_losses),
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

    def train_pairwise_v1_multi_task_v2(self, verbose=1, ddp=False):
        self.logger = self.config.logger
        train_start = time()

        num_epochs = self.config.train.pretrain_epochs

        if ddp and self.config.train.use_memory_efficient_dataset != 'v1':
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set)
            dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size, drop_last=True,
                                    collate_fn=dataset.collate_affinity_pair_wise_multi_task_v2,
                                    num_workers=self.config.train.num_workers,
                                    sampler=train_sampler)
        else:
            dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size, drop_last=True,
                                    shuffle=self.config.train.shuffle, collate_fn=dataset.collate_affinity_pair_wise_multi_task_v2,
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

            if ddp and self.config.train.use_memory_efficient_dataset != 'v1':
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
                               self.config.train.pretrain_mtl_Kd_lambda * ranking_loss_K

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
                batch_regression_k_losses.append(self.config.train.pretrain_mtl_Kd_lambda * regression_loss_K.item())

                batch_ranking_ic50_losses.append(self.config.train.pretrain_mtl_IC50_lambda * ranking_loss_IC50.item())
                batch_ranking_k_losses.append(self.config.train.pretrain_mtl_Kd_lambda * ranking_loss_K.item())

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

    def train_pointwise(self, verbose=1, ddp=False):
        self.logger = self.config.logger
        train_start = time()
        num_epochs = self.config.train.pretrain_epochs

        if ddp and self.config.train.use_memory_efficient_dataset != 'v1':
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set)
            dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size,
                                    collate_fn=dataset.collate_pdbbind_affinity,
                                    num_workers=self.config.train.num_workers, drop_last=True,
                                    sampler=train_sampler)
        else:
            dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size,
                                    shuffle=self.config.train.shuffle, collate_fn=dataset.collate_pdbbind_affinity,
                                    num_workers=self.config.train.num_workers, drop_last=True)

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
            batch_losses, batch_regression_losses = [], []

            if ddp and self.config.train.use_memory_efficient_dataset != 'v1':
                dataloader.sampler.set_epoch(epoch)

            for batch in dataloader:
                if self.device.type == "cuda":
                    batch = self.trans_device(batch)

                y_pred, x_output, _ = model(batch)
                regression_loss = self.loss_fn(y_pred, batch[-3])

                pretrain_loss = regression_loss

                if not pretrain_loss.requires_grad:
                    raise RuntimeError("loss doesn't require grad")

                self._optimizer.zero_grad()
                pretrain_loss.backward()
                self._optimizer.step()

                batch_losses.append(pretrain_loss.item())
                batch_regression_losses.append(regression_loss.item())

            train_losses.append(sum(batch_losses))

            if self.logger is not None:
                self.logger.info('Epoch: %d | Pretrain Loss: %.4f | Regression Loss: %.4f | Ranking Loss: %.4f | Lr: %.4f | Time: %.4f' % (
                epoch + start_epoch, sum(batch_losses), sum(batch_regression_losses), 0.0, self._optimizer.param_groups[0]['lr'], time() - epoch_start))

            if (not ddp) or (ddp and dist.get_rank() == 0):
                # evaluate
                if self.config.train.eval:
                    eval_acc, eval_rmse = self.evaluate_pointwise('val', verbose=1, logger=self.logger)
                    val_matric.append(eval_acc)

                if self.config.train.scheduler.type == "plateau":
                    self._scheduler.step(train_losses[-1])
                else:
                    self._scheduler.step()

                val_list = {
                    'cur_epoch': epoch + start_epoch,
                    'best_matric': best_matric,
                }

                self.save(self.config.train.save_path, 'latest', ddp, val_list)

                if sum(batch_losses) < best_loss:
                    best_loss = sum(batch_losses)
                    self.save(self.config.train.save_path, 'best_loss', ddp, val_list)

                if val_matric[-1] > best_matric:
                    early_stop = 0
                    best_matric = val_matric[-1]
                    if self.config.train.save:
                        print('saving checkpoint')
                        val_list = {
                            'cur_epoch': epoch + start_epoch,
                            'best_matric': best_matric,
                        }
                        self.save(self.config.train.save_path, epoch + start_epoch, ddp, val_list)

            torch.cuda.empty_cache()
            if epoch % self.config.train.pretrain_regression_loss_lambda_degrade_epoch == 0:
                self.config.train.pretrain_regression_loss_lambda *= self.config.train.pretrain_regression_loss_lambda_degrade_ratio

        self.best_matric = best_matric
        self.start_epoch = start_epoch + num_epochs
        print('optimization finished.')
        print('Total time elapsed: %.5fs' % (time() - train_start))

    def train_pointwise_multi_task(self, verbose=1, ddp=False):
        self.logger = self.config.logger
        train_start = time()
        num_epochs = self.config.train.pretrain_epochs

        if ddp and self.config.train.use_memory_efficient_dataset != 'v1':
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set)
            dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size,
                                    collate_fn=dataset.collate_pdbbind_affinity_multi_task,
                                    num_workers=self.config.train.num_workers, drop_last=True,
                                    sampler=train_sampler)
        else:
            dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size,
                                    shuffle=self.config.train.shuffle, collate_fn=dataset.collate_pdbbind_affinity_multi_task,
                                    num_workers=self.config.train.num_workers, drop_last=True)

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
            batch_losses, batch_regression_losses = [], []
            batch_regression_ic50_losses, batch_regression_kd_losses, batch_regression_ki_losses = [], [], []
            if ddp and self.config.train.use_memory_efficient_dataset != 'v1':
                dataloader.sampler.set_epoch(epoch)

            for batch in dataloader:
                if self.device.type == "cuda":
                    batch = self.trans_device(batch)

                (regression_loss_IC50, regression_loss_Kd, regression_loss_Ki), \
                (affinity_pred_IC50, affinity_pred_Kd, affinity_pred_Ki), \
                (affinity_IC50, affinity_Kd, affinity_Ki) = model(batch, ASRP=False)

                pretrain_loss = self.config.train.pretrain_mtl_IC50_lambda * regression_loss_IC50 + \
                                self.config.train.pretrain_mtl_Kd_lambda * regression_loss_Kd + \
                                self.config.train.pretrain_mtl_Ki_lambda * regression_loss_Ki

                if not pretrain_loss.requires_grad:
                    raise RuntimeError("loss doesn't require grad")

                self._optimizer.zero_grad()
                pretrain_loss.backward()
                self._optimizer.step()

                batch_losses.append(pretrain_loss.item())
                batch_regression_ic50_losses.append(self.config.train.pretrain_mtl_IC50_lambda * regression_loss_IC50.item())
                batch_regression_kd_losses.append(self.config.train.pretrain_mtl_Kd_lambda * regression_loss_Kd.item())
                batch_regression_ki_losses.append(self.config.train.pretrain_mtl_Ki_lambda * regression_loss_Ki.item())

            train_losses.append(sum(batch_losses))

            if self.logger is not None:
                self.logger.info('Epoch: %d | Pretrain Loss: %.4f | '
                                 'Regression IC50 Loss: %.4f | Regression Kd Loss: %.4f | Regression Ki Loss: %.4f | '
                                 'Lr: %.4f | Time: %.4f' % (
                epoch + start_epoch, sum(batch_losses),
                sum(batch_regression_ic50_losses), sum(batch_regression_kd_losses), sum(batch_regression_ki_losses),
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

    def train_pointwise_multi_task_v2(self, verbose=1, ddp=False):
        self.logger = self.config.logger
        train_start = time()
        num_epochs = self.config.train.pretrain_epochs

        if ddp and self.config.train.use_memory_efficient_dataset != 'v1':
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set)
            dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size,
                                    collate_fn=dataset.collate_pdbbind_affinity_multi_task_v2,
                                    num_workers=self.config.train.num_workers, drop_last=True,
                                    sampler=train_sampler)
        else:
            dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size,
                                    shuffle=self.config.train.shuffle, collate_fn=dataset.collate_pdbbind_affinity_multi_task_v2,
                                    num_workers=self.config.train.num_workers, drop_last=True)

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
            batch_losses, batch_regression_losses = [], []
            batch_regression_ic50_losses, batch_regression_k_losses = [], []
            if ddp and self.config.train.use_memory_efficient_dataset != 'v1':
                dataloader.sampler.set_epoch(epoch)

            for batch in dataloader:
                if self.device.type == "cuda":
                    batch = self.trans_device(batch)

                (regression_loss_IC50, regression_loss_K), \
                (affinity_pred_IC50, affinity_pred_K), \
                (affinity_IC50, affinity_K) = model(batch, ASRP=False)

                pretrain_loss = self.config.train.pretrain_mtl_IC50_lambda * regression_loss_IC50 + \
                                self.config.train.pretrain_mtl_K_lambda * regression_loss_K

                if not pretrain_loss.requires_grad:
                    raise RuntimeError("loss doesn't require grad")

                self._optimizer.zero_grad()
                pretrain_loss.backward()
                self._optimizer.step()

                batch_losses.append(pretrain_loss.item())
                batch_regression_ic50_losses.append(self.config.train.pretrain_mtl_IC50_lambda * regression_loss_IC50.item())
                batch_regression_k_losses.append(self.config.train.pretrain_mtl_K_lambda * regression_loss_K.item())

            train_losses.append(sum(batch_losses))

            if self.logger is not None:
                self.logger.info('Epoch: %d | Pretrain Loss: %.4f | '
                                 'Regression IC50 Loss: %.4f | Regression K Loss: %.4f | '
                                 'Lr: %.4f | Time: %.4f' % (
                epoch + start_epoch, sum(batch_losses),
                sum(batch_regression_ic50_losses), sum(batch_regression_k_losses),
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