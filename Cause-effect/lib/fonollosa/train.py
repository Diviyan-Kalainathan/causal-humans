"""
Cause-effect model training

"""

# Author: Jose A. R. Fonollosa <jarfo@yahoo.com>
#
# License: Apache, Version 2.0

import numpy as np
import pandas as pd
import sys

import numpy as np
import pandas as pd

import data_io
import estimator as ce
import util
import cPickle as pickle

MODEL = ce.CauseEffectSystemCombination
MODEL_PARAMS = {'weights':[0.383, 0.370, 0.247], 'n_jobs':-1}


def parse_dataframe(df):
    parse_cell = lambda cell: np.fromstring(cell, dtype=np.float, sep=" ")
    df = df.applymap(parse_cell)
    return df

def read_data(pairs_path, info_path):
    df_pairs = parse_dataframe(pd.read_csv(pairs_path, index_col="SampleID"))
    df_info = pd.read_csv(info_path, index_col="SampleID")
    features = pd.concat([df_pairs, df_info], axis=1)
    features_inverse = features.copy()
    features_inverse['A'] = features['B']
    features_inverse['A type'] = features['B type']
    features_inverse['B'] = features['A']
    features_inverse['B type'] = features['A type']
    original_index = np.array(zip(features.index, features.index)).flatten()
    features = pd.concat([features, features_inverse])
    features.index = range(0,len(features),2)+range(1,len(features),2)
    features.sort(inplace=True)
    features.index = original_index
    features.index.name = "SampleID"
    return features

def read_target(targets_path):

    df = pd.read_csv(targets_path, index_col="SampleID")

    # Duplicate training sequences exchanging 'A' with 'B'
    df_inverse = df.copy()
    df_inverse.Target = -df.Target

    original_index = np.array(zip(df.index, df.index)).flatten()
    df = pd.concat([df, df_inverse])
    df.index = range(0,len(df),2)+range(1,len(df),2)
    df.sort(inplace=True)
    df.index = original_index
    df.index.name = "SampleID"
    return df

def save_model(model, modelpath):

    pickle.dump(model, open(modelpath, "w"))


def train(datapairs, datapublicinfo, datatarget, model_path):

    model = MODEL(**MODEL_PARAMS)

    train_filter = None
    train_filter2 = None

    print("Reading in training data " + datapairs[0])
    train = read_data(datapairs[0], datapublicinfo[0])
    print("Extracting features")
    train = model.extract(train)
    target = read_target(datatarget[0])

    train2 = None
    target2 = None

    for idx in range(1,len(datapairs)):
        print "Reading in training data", datapairs[idx]
        tr = read_data(datapairs[idx],datapublicinfo[idx])
        print "Extracting features"
        tr = model.extract(tr)

        tg = read_target(datatarget[idx])

        train2 = tr if train2 is None else pd.concat((train2, tr), ignore_index=True)
        target2 = tg if target2 is None else pd.concat((target2, tg), ignore_index=True)
        train2, target2 = util.random_permutation(train2, target2)
        train_filter2 = None

    # Data selection
    train, target = util.random_permutation(train, target)
    train_filter = None

    if train_filter is not None:
        train = train[train_filter]
        target = target[train_filter]
    if train_filter2 is not None:
        train2 = train2[train_filter2]
        target2 = target2[train_filter2]

    print("Training model with optimal weights")
    X = pd.concat([train, train2]) if train2 is not None else train
    y = np.concatenate((target.Target.values, target2.Target.values)) if target2 is not None else target.Target.values
    model.fit(X, y)

    print "Saving model", model_path
    save_model(model, model_path)


def main():
    
    set1 = 'train' if len(sys.argv) < 2 else sys.argv[1]
    set2 = [] if len(sys.argv) < 3 else sys.argv[2:]
    train_filter = None
    train_filter2 = None
    
    model = MODEL(**MODEL_PARAMS)
    
    print("Reading in training data " + set1)
    train = data_io.read_data(set1)
    print("Extracting features")
    train = model.extract(train)
    print("Saving train features")
    data_io.write_data(set1, train)
    target = data_io.read_target(set1)
    
    train2 = None
    target2 = None
    for s in set2:
        print "Reading in training data", s
        tr = data_io.read_data(s)
        print "Extracting features"
        tr = model.extract(tr)
        print "Saving train features"
        data_io.write_data(s, tr)
        tg = data_io.read_target(s)
        train2 = tr if train2 is None else pd.concat((train2, tr), ignore_index=True)
        target2 = tg if target2 is None else pd.concat((target2, tg), ignore_index=True)
        train2, target2 = util.random_permutation(train2, target2)
        train_filter2 = None

    # Data selection
    train, target = util.random_permutation(train, target)
    train_filter = None

    if train_filter is not None:
        train = train[train_filter]
        target = target[train_filter]
    if train_filter2 is not None:
        train2 = train2[train_filter2]
        target2 = target2[train_filter2]

    print("Training model with optimal weights")
    X = pd.concat([train, train2]) if train2 is not None else train
    y = np.concatenate((target.Target.values, target2.Target.values)) if target2 is not None else target.Target.values  
    model.fit(X, y)
    model_path = "model.pkl"
    print "Saving model", model_path
    data_io.save_model(model, model_path)

if __name__=="__main__":
    main()
