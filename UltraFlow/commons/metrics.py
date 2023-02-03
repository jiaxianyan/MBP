from scipy import stats
import torch
import torch.nn as nn
import numpy as np
from math import sqrt, ceil
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ndcg_score, recall_score
import os
import pickle
import dgl
from typing import Union, List
from torch import Tensor
from statistics import stdev

def affinity_loss(affinity_pred,labels,sec_pred,bg_prot,config):
    loss = nn.MSELoss(affinity_pred,labels)
    if config.model.aux_w != 0:
        loss += config.train.aux_w * nn.CrossEntropyLoss(sec_pred,bg_prot.ndata['s'])
    return loss

def Accurate_num(outputs,y):
    _, y_pred_label = torch.max(outputs, dim=1)
    return torch.sum(y_pred_label == y.data).item()

def RMSE(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def MAE(y,f):
    mae = (np.abs(y-f)).mean()
    return mae

def SD(y,f):
    f,y = f.reshape(-1,1),y.reshape(-1,1)
    lr = LinearRegression()
    lr.fit(f,y)
    y_ = lr.predict(f)
    sd = (((y - y_) ** 2).sum() / (len(y) - 1)) ** 0.5
    return sd

def Pearson(y,f):
    y,f = y.flatten(),f.flatten()
    rp = np.corrcoef(y, f)[0,1]
    return rp

def Spearman(y,f):
    y, f = y.flatten(), f.flatten()
    rp = stats.spearmanr(y, f)
    return rp[0]

def NDCG(y,f,k=None):
    y, f = y.flatten(), f.flatten()
    return ndcg_score(np.expand_dims(y, axis=0), np.expand_dims(f,axis=0),k=k)

def Recall(y, f, postive_threshold = 7.5):
    y, f = y.flatten(), f.flatten()
    y_class = y > postive_threshold
    f_class = f > postive_threshold

    return recall_score(y_class, f_class)

def Enrichment_Factor(y, f, postive_threshold = 7.5, top_percentage = 0.001):
    y, f = y.flatten(), f.flatten()
    y_class = y > postive_threshold
    f_class = f > postive_threshold

    data = list(zip(y_class.tolist(), f_class.tolist()))
    data.sort(key=lambda x:x[1], reverse=True)

    y_class, f_class = map(list, zip(*data))

    total_active_rate = sum(y_class) / len(y_class)
    top_num = ceil(len(y_class) * top_percentage)
    top_active_rate = sum(y_class[:top_num]) / top_num

    er = top_active_rate / total_active_rate

    return er

def Auxiliary_Weight_Balance(aux_type='Q8'):
    if os.path.exists('loss_weight.pkl'):
        with open('loss_weight.pkl','rb') as f:
            w = pickle.load(f)
        return w[aux_type]

def RMSD(ligs_coords_pred, ligs_coords):
    rmsds = []
    for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
        rmsds.append(torch.sqrt(torch.mean(torch.sum(((lig_coords_pred - lig_coords) ** 2), dim=1))).item())
    return rmsds

def KabschRMSD(ligs_coords_pred, ligs_coords):
    rmsds = []
    for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
        lig_coords_pred_mean = lig_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
        lig_coords_mean = lig_coords.mean(dim=0, keepdim=True)  # (1,3)

        A = (lig_coords_pred - lig_coords_pred_mean).transpose(0, 1) @ (lig_coords - lig_coords_mean)

        U, S, Vt = torch.linalg.svd(A)

        corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
        rotation = (U @ corr_mat) @ Vt
        translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)

        lig_coords = (rotation @ lig_coords.t()).t() + translation
        rmsds.append(torch.sqrt(torch.mean(torch.sum(((lig_coords_pred - lig_coords) ** 2), dim=1))))
    return torch.tensor(rmsds).mean()


class RMSDmedian(nn.Module):
    def __init__(self) -> None:
        super(RMSDmedian, self).__init__()

    def forward(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
        rmsds = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            rmsds.append(torch.sqrt(torch.mean(torch.sum(((lig_coords_pred - lig_coords) ** 2), dim=1))))
        return torch.median(torch.tensor(rmsds))


class RMSDfraction(nn.Module):
    def __init__(self, distance) -> None:
        super(RMSDfraction, self).__init__()
        self.distance = distance

    def forward(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
        rmsds = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            rmsds.append(torch.sqrt(torch.mean(torch.sum(((lig_coords_pred - lig_coords) ** 2), dim=1))))
        count = torch.tensor(rmsds) < self.distance
        return 100 * count.sum() / len(count)


def CentroidDist(ligs_coords_pred, ligs_coords):
    distances = []
    for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            distances.append(torch.linalg.norm(lig_coords_pred.mean(dim=0)-lig_coords.mean(dim=0)).item())
    return distances


class CentroidDistMedian(nn.Module):
    def __init__(self) -> None:
        super(CentroidDistMedian, self).__init__()

    def forward(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
        distances = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            distances.append(torch.linalg.norm(lig_coords_pred.mean(dim=0)-lig_coords.mean(dim=0)))
        return torch.median(torch.tensor(distances))


class CentroidDistFraction(nn.Module):
    def __init__(self, distance) -> None:
        super(CentroidDistFraction, self).__init__()
        self.distance = distance

    def forward(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
        distances = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            distances.append(torch.linalg.norm(lig_coords_pred.mean(dim=0)-lig_coords.mean(dim=0)))
        count = torch.tensor(distances) < self.distance
        return 100 * count.sum() / len(count)

class MeanPredictorLoss(nn.Module):

    def __init__(self, loss_func) -> None:
        super(MeanPredictorLoss, self).__init__()
        self.loss_func = loss_func

    def forward(self, x1: Tensor, targets: Tensor) -> Tensor:
        return self.loss_func(torch.full_like(targets, targets.mean()), targets)


def compute_mmd(source, target, batch_size=1000, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Calculate the `maximum mean discrepancy distance <https://jmlr.csail.mit.edu/papers/v13/gretton12a.html>`_ between two sample set.
    This implementation is based on `this open source code <https://github.com/ZongxianLee/MMD_Loss.Pytorch>`_.
    Args:
        source (pytorch tensor): the pytorch tensor containing data samples of the source distribution.
        target (pytorch tensor): the pytorch tensor containing data samples of the target distribution.
    :rtype:
        :class:`float`
    """
    n_source = int(source.size()[0])
    n_target = int(target.size()[0])
    n_samples = n_source + n_target

    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0)
    total1 = total.unsqueeze(1)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth, id = 0.0, 0
        while id < n_samples:
            bandwidth += torch.sum((total0 - total1[id:id + batch_size]) ** 2)
            id += batch_size
        bandwidth /= n_samples ** 2 - n_samples

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    XX_kernel_val = [0 for _ in range(kernel_num)]
    for i in range(kernel_num):
        XX_kernel_val[i] += torch.sum(
            torch.exp(-((total0[:, :n_source] - total1[:n_source, :]) ** 2) / bandwidth_list[i]))
    XX = sum(XX_kernel_val) / (n_source * n_source)

    YY_kernel_val = [0 for _ in range(kernel_num)]
    id = n_source
    while id < n_samples:
        for i in range(kernel_num):
            YY_kernel_val[i] += torch.sum(
                torch.exp(-((total0[:, n_source:] - total1[id:id + batch_size, :]) ** 2) / bandwidth_list[i]))
        id += batch_size
    YY = sum(YY_kernel_val) / (n_target * n_target)

    XY_kernel_val = [0 for _ in range(kernel_num)]
    id = n_source
    while id < n_samples:
        for i in range(kernel_num):
            XY_kernel_val[i] += torch.sum(
                torch.exp(-((total0[:, id:id + batch_size] - total1[:n_source, :]) ** 2) / bandwidth_list[i]))
        id += batch_size
    XY = sum(XY_kernel_val) / (n_source * n_target)

    return XX.item() + YY.item() - 2 * XY.item()


def get_matric_dict(rmsds, centroids, kabsch_rmsds=None):
    rmsd_mean = sum(rmsds)/len(rmsds)
    centroid_mean = sum(centroids) / len(centroids)
    rmsd_std = stdev(rmsds)
    centroid_std = stdev(centroids)

    # rmsd < 2
    count = torch.tensor(rmsds) < 2.0
    rmsd_less_than_2 = 100 * count.sum().item() / len(count)

    # rmsd < 2
    count = torch.tensor(rmsds) < 5.0
    rmsd_less_than_5 = 100 * count.sum().item() / len(count)

    # centorid < 2
    count = torch.tensor(centroids) < 2.0
    centroid_less_than_2 = 100 * count.sum().item() / len(count)

    # centorid < 5
    count = torch.tensor(centroids) < 5.0
    centroid_less_than_5 = 100 * count.sum().item() / len(count)

    rmsd_precentiles = np.percentile(np.array(rmsds), [25, 50, 75]).round(4)
    centroid_prcentiles = np.percentile(np.array(centroids), [25, 50, 75]).round(4)

    metrics_dict = {'rmsd mean': rmsd_mean, 'rmsd std': rmsd_std,
                    'rmsd 25%': rmsd_precentiles[0], 'rmsd 50%': rmsd_precentiles[1], 'rmsd 75%': rmsd_precentiles[2],
                    'centroid mean': centroid_mean, 'centroid std': centroid_std,
                    'centroid 25%': centroid_prcentiles[0], 'centroid 50%': centroid_prcentiles[1], 'centroid 75%': centroid_prcentiles[2],
                    'rmsd less than 2': rmsd_less_than_2, 'rmsd less than 5':rmsd_less_than_5,
                    'centroid less than 2': centroid_less_than_2, 'centroid less than 5': centroid_less_than_5,
                    }

    if kabsch_rmsds is not None:
        kabsch_rmsd_mean = sum(kabsch_rmsds) / len(kabsch_rmsds)
        kabsch_rmsd_std = stdev(kabsch_rmsd_mean)
        metrics_dict['kabsch rmsd mean'] = kabsch_rmsd_mean
        metrics_dict['kabsch rmsd std'] = kabsch_rmsd_std

    return metrics_dict

def get_sbap_regression_metric_dict(np_y, np_f):
    rmse, mae, pearson, spearman, sd_ = RMSE(np_y, np_f), \
                                   MAE(np_y, np_f),\
                                   Pearson(np_y,np_f), \
                                   Spearman(np_y, np_f),\
                                   SD(np_y, np_f)

    metrics_dict = {'RMSE': rmse, 'MAE': mae, 'Pearson': pearson, 'Spearman': spearman, 'SD':sd_}
    return metrics_dict

def get_sbap_matric_dict(np_y, np_f):
    rmse, mae, pearson, spearman, sd_ = RMSE(np_y, np_f), \
                                   MAE(np_y, np_f),\
                                   Pearson(np_y,np_f), \
                                   Spearman(np_y, np_f),\
                                   SD(np_y, np_f)

    recall, ndcg = Recall(np_y, np_f), NDCG(np_y, np_f)
    enrichment_factor = Enrichment_Factor(np_y, np_f)

    metrics_dict = {'RMSE': rmse, 'MAE': mae, 'Pearson': pearson, 'Spearman': spearman, 'SD':sd_,
                    'Recall': recall, 'NDCG': ndcg, 'EF1%':enrichment_factor
                    }
    return metrics_dict

def get_matric_output_str(matric_dict):
    matric_str = ''
    for key in matric_dict.keys():
        if not 'less than' in key:
            matric_str += '| {}: {:.4f} '.format(key, matric_dict[key])
        else:
            matric_str += '| {}: {:.4f}% '.format(key, matric_dict[key])
    return matric_str

def get_unseen_matric(rmsds, centroids, names, unseen_file_path):
    with open(unseen_file_path, 'r') as f:
        unseen_names = f.read().strip().split('\n')
    unseen_rmsds, unseen_centroids = [], []
    for name, rmsd, centroid in zip(names, rmsds, centroids):
        if name in unseen_names:
            unseen_rmsds.append(rmsd)
            unseen_centroids.append(centroid)
    return get_matric_dict(unseen_rmsds, unseen_centroids)