import numpy as np
from scipy.optimize import linear_sum_assignment


def cluster_acc(Y_pred, Y):
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
        
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / Y_pred.size
