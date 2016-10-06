

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


def score(y,p):
  return (roc_auc_score(y==1,p)+roc_auc_score(y==-1,-p))/2

def evalScore(benchmarkpath,benchmarkname, algo,targetpath,privateinfopath ):



    dfresultsGlobal = pd.read_csv(algo + "_" + benchmarkname[0], index_col="SampleID")
    dftargetGlobal = pd.read_csv(targetpath[0], index_col="SampleID")
    dfprivateinfoGlobal = pd.read_csv(privateinfopath[0], index_col="SampleID")



    for i in range(1,len(benchmarkname)):

        dfresults = pd.read_csv(algo + "_" + benchmarkname[i], index_col="SampleID")
        dftarget = pd.read_csv(targetpath[i], index_col="SampleID")
        dfprivateinfo = pd.read_csv(privateinfopath[i], index_col="SampleID")

        dfresultsGlobal.append(dfresults)
        dftargetGlobal.append(dftarget)
        dfprivateinfoGlobal.append(dfprivateinfo)


    dfresultsGlobal["Final target"] = dftargetGlobal["Target"]
    dfresultsGlobal["Source"] = dftargetGlobal["Source"]
    dfresultsGlobal["A type"] = dftargetGlobal["A type"]
    dfresultsGlobal["B type"] = dftargetGlobal["A type"]
    dfresultsGlobal["sample num"] = dftargetGlobal["sample num"]
    dfresultsGlobal["RealData [yes=1]"] = dftargetGlobal["RealData [yes=1]"]

    listdf = []
    listdf.append(dfresultsGlobal[dfresultsGlobal["RealData [yes=1]"] == 1])
    listdf.append(dfresultsGlobal[dfresultsGlobal["RealData [yes=1]"] == 0])

    listdf.append(dfresultsGlobal[dfresultsGlobal["A type"] == "Numerical" and dfresultsGlobal["B type"] == "Numerical"])
    listdf.append(dfresultsGlobal[dfresultsGlobal["A type"] == "Categorical" and dfresultsGlobal["B type"] == "Categorical"])
    listdf.append(dfresultsGlobal[dfresultsGlobal["A type"] == "Binary" and dfresultsGlobal["B type"] == "Binary"])

    listdf.append(dfresultsGlobal[(dfresultsGlobal["A type"] == "Numerical" and dfresultsGlobal["B type"] == "Categorical") or (dfresultsGlobal["B type"] == "Numerical" and dfresultsGlobal["A type"] == "Categorical")])

    listdf.append(dfresultsGlobal[(dfresultsGlobal["A type"] == "Numerical" and dfresultsGlobal["B type"] == "Binary")
                                  or ( dfresultsGlobal["B type"] == "Numerical" and dfresultsGlobal["A type"] == "Binary")])

    listdf.append(dfresultsGlobal[(dfresultsGlobal["A type"] == "Categorical" and dfresultsGlobal["B type"] == "Binary")
                                  or ( dfresultsGlobal["B type"] == "Categorical" and dfresultsGlobal["A type"] == "Binary")])


    listNumSample = [100,200,300,400,500,600,700,800,900,1000,1500,2000,3000]


    for df in listdf:
        ROCscore = score(df["Target"], df["Final target"])
        numPair = df["Target"].count



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




