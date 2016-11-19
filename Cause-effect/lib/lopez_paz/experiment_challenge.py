"""
Code made by David Lopez-Paz
All credits to Mr. Lopez-Paz
"""
import multiprocessing as mp
import pandas as pd
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

def rp(k, s, d):
    return np.hstack((np.vstack([si * np.random.randn(k, d) for si in s]),
                      2 * np.pi * np.random.rand(k * len(s), 1))).T

try:
    param_defined
except NameError:
    param_defined=True
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



def task_featurize(inputdata,outputdata):
    # print("featurize test ")
    print("OK")
    x_te = featurize(inputdata)
    np.savetxt( outputdata, x_te)



def featurizeData(data, outputdata, max_proc):

    pool = mp.Pool(processes=max_proc)

    for idx in range(len(data)):
        print('Data ' + str(idx))
        pool.apply_async(task_featurize, args=(data[idx], outputdata[idx],))
    pool.close()
    pool.join()


def train(featurizedData, targetData, nameModel):


    for idx in range(len(featurizedData)):
        print('Data ' + str(idx))
        x_tr_temp = np.loadtxt(featurizedData[idx])

        dfTarget = pd.read_csv(targetData[idx], sep = ",")
        y_tr_temp = dfTarget["Target"].values

        y_tr_temp = np.hstack((y_tr_temp, -y_tr_temp))

        if(idx == 0):
            x_tr = x_tr_temp
            y_tr = y_tr_temp
        else :
            x_tr = np.vstack((x_tr,x_tr_temp))
            y_tr = np.hstack((y_tr, y_tr_temp))

    x_ab = x_tr[(y_tr == 1) | (y_tr == -1)]
    y_ab = y_tr[(y_tr == 1) | (y_tr == -1)]


    params = {'random_state': 0, 'n_estimators': E, 'min_samples_leaf': L, 'n_jobs': 8}

    print("start learning ")

    clf0 = CLF(**params).fit(x_tr, y_tr != 0)  # causal or confounded?
    clf1 = CLF(**params).fit(x_ab, y_ab == 1)  # causal or anticausal?

    pickle.dump(clf0, open(nameModel + "0.p", "wb"))
    pickle.dump(clf1, open(nameModel + "1.p", "wb"))




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
    sys.stdout.flush()


def predict(data,results,modelPath, max_proc):

    print("start predict ")

    clf0 = pickle.load(open(modelPath + "clf0.p", "rb"))
    clf1 = pickle.load(open(modelPath + "clf1.p", "rb"))
    pool = mp.Pool(processes=max_proc)

    for idx in range(len(data)):
        print('Data '+str(idx))
        pool.apply_async(task_pred,args=(data[idx],results[idx],clf0,clf1,))
    pool.close()
    pool.join()


