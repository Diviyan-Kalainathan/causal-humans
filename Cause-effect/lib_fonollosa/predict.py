"""
Cause-effect direction prediction using the model saved in model.pkl

"""

# Author: Jose A. R. Fonollosa <jarfo@yahoo.com>
#
# License: Apache, Version 2.0

import sys
import csv
import numpy as np
import pandas as pd
import cPickle as pickle
from   sklearn.metrics       import roc_auc_score
import sys
import estimator
import features

def load_model(model_path, verbose=True):


    sys.modules['estimator'] = estimator
    sys.modules['features'] = features

    m = pickle.load(open(model_path))
    return m

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

def write_predictions(pred_path, test, predictions):
    writer = csv.writer(open(pred_path, "w"), lineterminator="\n")
    rows = [x for x in zip(test.index, predictions)]
    writer.writerow(("SampleID", "Target"))
    writer.writerows(rows)

def score(y,p):
  return (roc_auc_score(y==1,p)+roc_auc_score(y==-1,-p))/2


def predict(pairs_path, info_path, result_path, model_path):


    test = read_data(pairs_path, info_path)

    print "Loading the classifier"

    print(model_path + "model.pkl")

    model = load_model(model_path + "model.pkl")
    print "model.weights", model.weights

    print "Extracting features"
    test = model.extract(test)

    print "Making predictions"
    predictions = model.predict(test)
    print "Writing predictions to file"
    write_predictions(result_path, test[0::2], predictions[0::2])









