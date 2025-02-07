from sklearn import metrics
from math import sqrt
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

def accuracy(y_pred, y_true):
    """
    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred (numpy.array):
        y_true (numpy.array): [description]
    """
    return metrics.accuracy_score(y_pred=y_pred.round(), y_true=y_true)

def precision(y_pred, y_true):
    """
    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred (numpy.array):
        y_true (numpy.array): [description]
    """
    return metrics.precision_score(y_pred=y_pred.round(), y_true=y_true)

def recall(y_pred, y_true):
    """
    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred (numpy.array):[description]
        y_true (numpy.array): [description]
    Returns:
        [type]: [description]
    """
    return metrics.recall_score(y_pred=y_pred.round(), y_true=y_true)

def f1_score(y_pred, y_true):
    return metrics.f1_score(y_pred=y_pred.round(), y_true=y_true)

def bacc_score(y_pred, y_true):
    return metrics.balanced_accuracy_score(y_pred=y_pred.round(), y_true=y_true)

def roc_auc(y_pred, y_true):
    """
    The values of y_pred can be decimicals, within 0 and 1.
    Args:
        y_pred ([type]): [description]
        y_true ([type]): [description]

    Returns:
        [type]: [description]
    """
    return metrics.roc_auc_score(y_score=y_pred, y_true=y_true)

def mcc_score(y_pred, y_true):
    return metrics.matthews_corrcoef(y_pred=y_pred.round(), y_true=y_true)

def kappa(y_pred, y_true):
    return metrics.cohen_kappa_score(y2=y_pred.round(), y1=y_true)

def ap_score(y_pred, y_true):
    """
    The values of y_pred can be decimicals, within 0 and 1.
    Args:
        y_pred ([type]): [description]
        y_true ([type]): [description]

    Returns:
        [type]: [description]
    """
    return metrics.average_precision_score(y_score=y_pred, y_true=y_true)


def rmse(y, f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def mse(y, f):
    mse = ((y - f)**2).mean(axis=0)
    return mse

def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp

def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def R2(y,f):
    r2=r2_score(y, f)
    return r2
