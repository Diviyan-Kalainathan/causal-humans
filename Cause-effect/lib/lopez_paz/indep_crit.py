"""
Code made by David Lopez-Paz
All credits to Mr. Lopez-Paz
"""
import multiprocessing as mp

import numpy as np
import sys
import cPickle as pickle
import pandas as pd

from   sklearn.preprocessing import scale
from   sklearn.ensemble import RandomForestClassifier     as CLF
from   sklearn.metrics import roc_auc_score

def rp(k, s, d):
    return np.hstack((np.vstack([si * np.random.randn(k, d) for si in s]),
                      2 * np.pi * np.random.rand(k * len(s), 1))).T

def f1(x, w):
    return np.cos(np.dot(np.hstack((x, np.ones((x.shape[0], 1)))), w))


class lp_indep_criterion:

    def __init__(self):
        with open('clfindep.p','rb') as paramfile:
            self.clfd = pickle.load(paramfile)
        np.random.seed(0)

        # K = int(sys.argv[1])
        # E = int(sys.argv[2])
        # L = int(sys.argv[3])
        K = 333  # Nb of features/3

        E = 500  # Nb of trees in random forest
        L = 20  # Nb of min leaves

        self.wx = rp(K, [0.15, 1.5, 15], 1)
        self.wy = rp(K, [0.15, 1.5, 15], 1)
        self.wz = rp(K, [0.15, 1.5, 15], 2)



    def predict_indep(self,X,Y):
        x = scale(np.array(X))[:, np.newaxis]
        y = scale(np.array(Y))[:, np.newaxis]
        d = np.hstack((f1(x, self.wx).mean(0), f1(y, self.wy).mean(0), f1(np.hstack((x, y)), self.wz).mean(0)))

        return float(self.clfd.predict_proba(d)[:,1])
