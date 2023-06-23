from scipy import stats
import torch
import torch.nn as nn
import numpy as np
from math import sqrt, ceil
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ndcg_score, recall_score

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

def get_sbap_regression_metric_dict(np_y, np_f):
    rmse, mae, pearson, spearman, sd_ = RMSE(np_y, np_f), \
                                   MAE(np_y, np_f),\
                                   Pearson(np_y,np_f), \
                                   Spearman(np_y, np_f),\
                                   SD(np_y, np_f)

    metrics_dict = {'RMSE': rmse, 'MAE': mae, 'Pearson': pearson, 'Spearman': spearman, 'SD':sd_}
    return metrics_dict

def get_matric_output_str(matric_dict):
    matric_str = ''
    for key in matric_dict.keys():
        if not 'less than' in key:
            matric_str += '| {}: {:.4f} '.format(key, matric_dict[key])
        else:
            matric_str += '| {}: {:.4f}% '.format(key, matric_dict[key])
    return matric_str