"""
Code made by David Lopez-Paz
All credits to Mr. Lopez-Paz
"""
import multiprocessing as mp
F_X_TR = "data/pairs.csv"
F_Y_TR = "data/targets.csv"

# F_X_TE = "data/CE_pairs_secafi_correl_pvalue_5percent.csv"

#F_X_TE = "data/pairs_c_6_p0.csv"

# F_X_TE = "data/test_pair.csv"


# F_Y_TE = "/data/test_target.csv"

import numpy as np
import sys
import cPickle as pickle
import pandas as pd

from   sklearn.preprocessing import scale
from   sklearn.ensemble import RandomForestClassifier     as CLF
from   sklearn.metrics import roc_auc_score


# from   sklearn.grid_search   import GridSearchCV
# from   scipy.stats           import skew, kurtosis, rankdata

def rp(k, s, d):
    return np.hstack((np.vstack([si * np.random.randn(k, d) for si in s]),
                      2 * np.pi * np.random.rand(k * len(s), 1))).T


def f1(x, w):
    return np.cos(np.dot(np.hstack((x, np.ones((x.shape[0], 1)))), w))


def score(y, p):
    return (roc_auc_score(y == 1, p) + roc_auc_score(y == -1, -p)) / 2


def featurize_row(row, i, j):
    r = row.split(",", 2)

    x = scale(np.array(r[i].split(), dtype=np.float))[:, np.newaxis]
    y = scale(np.array(r[j].split(), dtype=np.float))[:, np.newaxis]

    d = np.hstack((f1(x, wx).mean(0), f1(y, wy).mean(0), f1(np.hstack((x, y)), wz).mean(0)))
    return d


def parse_dataframe(df):
    parse_cell = lambda cell: np.fromstring(cell, dtype=np.float, sep=" ")
    df = df.applymap(parse_cell)
    return df


def featurize(filename):
    # df = parse_dataframe(pd.read_csv(filename, index_col="SampleID"))
    f = open(filename);
    pairs = f.readlines();
    pairs.pop(0)
    print(len(pairs))
    f.close();
    return np.vstack((np.array([featurize_row(row, 1, 2) for row in pairs]),
                      np.array([featurize_row(row, 2, 1) for row in pairs])))


def featurizeTest(filename):
    # df = parse_dataframe(pd.read_csv(filename, index_col="SampleID"))
    f = open(filename);
    pairs = f.readlines();
    pairs.pop(0)
    print(len(pairs))
    f.close();
    # return np.vstack((np.array([featurize_row(row,1,2) for row in pairs]),
    #                   np.array([featurize_row(row,2,1) for row in pairs])))

    return np.vstack((np.array([featurize_row(row, 1, 2) for row in pairs])))


np.random.seed(0)

# K = int(sys.argv[1])
# E = int(sys.argv[2])
# L = int(sys.argv[3])

K = 333 #Nb of features/3

E = 500 #Nb of trees in random forest
L = 20 #Nb of min leaves

wx = rp(K, [0.15, 1.5, 15], 1)
wy = rp(K, [0.15, 1.5, 15], 1)
wz = rp(K, [0.15, 1.5, 15], 2)

print("wx " + str(wx.shape))
print("wy " + str(wy.shape))
print("wz " + str(wz.shape))
'''
print("featurize ")

x_tr = featurize(F_X_TR)

print("save features  ")

np.savetxt("features.csv", x_tr)'''#Already trained

'''print("load features  ")

x_tr = np.loadtxt("features.csv")

y_tr = np.genfromtxt(F_Y_TR, delimiter=",")

# y_te = np.genfromtxt(F_Y_TE, delimiter=",")[:,1]
# d_tr = (np.genfromtxt(F_Y_TR, delimiter=",")[:,2])==4
# d_te = (np.genfromtxt(F_Y_TE, delimiter=",")[:,2])==4

# pickle.dump(x_tr, open("x_tr", "w"))
# pickle.dump(y_tr, open("y_tr", "w"))
#
# x_tr = pickle.load(open("x_tr"))
# y_tr = pickle.load(open("y_tr"))

y_tr = np.hstack((y_tr, -y_tr))

# y_te = np.hstack((y_te,-y_te))
# d_tr = np.hstack((d_tr,d_tr))
# d_te = np.hstack((d_te,d_te))


x_ab = x_tr[(y_tr == 1) | (y_tr == -1)]
y_ab = y_tr[(y_tr == 1) | (y_tr == -1)]

params = {'random_state': 0, 'n_estimators': E, 'max_features': None,
          'max_depth': 50, 'min_samples_leaf': 10, 'verbose': 10}

params = {'random_state': 0, 'n_estimators': E, 'min_samples_leaf': L, 'n_jobs': 8}

print("start learning ")

clf0 = CLF(**params).fit(x_tr, y_tr != 0)  # causal or confounded?
clf1 = CLF(**params).fit(x_ab, y_ab == 1)  # causal or anticausal?
# clfd = CLF(**params).fit(x_tr,d_tr)    # dependent or independent?

pickle.dump(clf0, open("clf0.p", "wb"))
pickle.dump(clf1, open("clf1.p", "wb"))'''#Already trained

def task_pred(inputdata,outputdata,clf0,clf1):
    # print("featurize test ")
    x_te = featurizeTest(inputdata)
    p_te = clf0.predict_proba(x_te)[:, 1] * (2 * clf1.predict_proba(x_te)[:, 1] - 1)

    # print(p_te)

    df = pd.read_csv(inputdata, index_col="SampleID")

    Results = pd.DataFrame(index=df.index)

    Results['Target'] = p_te
    Results.to_csv(outputdata, sep=';', encoding='utf-8')
    sys.stdout.write('Generated output file '+ outputdata)

def predict(data,results,max_proc):
    print("start predict ")

    clf0 = pickle.load(open("clf0.p", "rb"))
    clf1 = pickle.load(open("clf1.p", "rb"))
    pool = mp.Pool(processes=max_proc)

    for idx in range(len(data)):
        print('Data '+str(idx))
        pool.apply_async(task_pred,args=(data[idx],results[idx],clf0,clf1,))
    pool.close()
    pool.join()

    '''x_te = featurizeTest(data[idx])
    p_te = clf0.predict_proba(x_te)[:, 1] * (2 * clf1.predict_proba(x_te)[:, 1] - 1)

    #print(p_te)

    df = pd.read_csv(data[idx], index_col="SampleID")

    Results = pd.DataFrame(index=df.index)

    Results['Target'] = p_te
    Results.to_csv(results[idx], sep=';', encoding='utf-8')'''


# print([score(y_te,p_te),clf0.score(x_te,y_te!=0),clfd.score(x_te,d_te)])
