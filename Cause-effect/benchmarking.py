

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


def score(y,p):
  return (roc_auc_score(y==1,p)+roc_auc_score(y==-1,-p))/2

def evalScore(benchmarkpath,benchmarkname, algo,targetpath,privateinfopath ):

    ROCscore = []

    resultsGlobal =



    for i in range(len(benchmarkname)):

        dfresults = pd.read_csv(algo + "_" + benchmarkname[i], index_col="SampleID")
        dftarget = pd.read_csv(targetpath[i], index_col="SampleID")
        dfprivateinfo = pd.read_csv(privateinfopath[i], index_col="SampleID")

        ROCscore.append(score(dfresults["Target"], dftarget["Target"]))



if __name__=="__main__":

    benchmarkpath = "output/benchmark/"

    resultpath = []
    targetpath = []
    privateinfopath = []
    benchmarkname = []

    benchmarkname.append("SUP3")
    targetpath.append("datacauseeffect/CEpairs/SUP3/CEdata_train_target.csv")
    privateinfopath.append("datacauseeffect/CEpairs/SUP3/CEdata_train_privateinfo.csv")

    benchmarkname.append("SUP4")
    targetpath.append("datacauseeffect/CEpairs/SUP4/CEnovel_test_target.csv")
    privateinfopath.append("datacauseeffect/CEpairs/SUP4/CEnovel_test_privateinfo.csv")

    benchmarkname.append("CEdata")
    targetpath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_target.csv")
    privateinfopath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_privateinfo.csv")

    for algo in ["Fonolossa", "LopezKernel" ]:

        evalScore(benchmarkpath,benchmarkname, algo,targetpath,privateinfopath )




