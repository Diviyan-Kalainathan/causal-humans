

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


def score(y,p):
  return (roc_auc_score(y==1,p)+roc_auc_score(y==-1,-p))/2

def scoreAcauseB(y,p):
  return roc_auc_score(y==1,p)

def scoreBcauseA(y,p):
  return roc_auc_score(y==-1,-p)


def evalScore(benchmarkpath,benchmarkname, algo,targetpath,privateinfopath ):


    dfresultsGlobal = pd.read_csv(benchmarkpath + algo + "_" + benchmarkname[0] + ".csv", index_col="SampleID")
    dftargetGlobal = pd.read_csv(targetpath[0], index_col="SampleID")
    dfprivateinfoGlobal = pd.read_csv(privateinfopath[0], index_col="SampleID")



    for i in range(1,len(benchmarkname)):

        dfresults = pd.read_csv(benchmarkpath + algo + "_" + benchmarkname[i] + ".csv", index_col="SampleID",sep = ';')
        dftarget = pd.read_csv(targetpath[i], index_col="SampleID")
        dfprivateinfo = pd.read_csv(privateinfopath[i], index_col="SampleID")

        dfresultsGlobal.append(dfresults)
        dftargetGlobal.append(dftarget)
        dfprivateinfoGlobal.append(dfprivateinfo)


    dfresultsGlobal["Final target"] = dftargetGlobal["Target"]
    dfresultsGlobal["Source"] = dfprivateinfoGlobal["Source"]
    dfresultsGlobal["A type"] = dfprivateinfoGlobal["A type"]
    dfresultsGlobal["B type"] = dfprivateinfoGlobal["A type"]
    dfresultsGlobal["sample num"] = dfprivateinfoGlobal["sample num"]
    dfresultsGlobal["RealData [yes=1]"] = dfprivateinfoGlobal["RealData [yes=1]"]

    listdf_type = []
    list_type = ["All data", "Real data","Artificial data" ]

    listdf_type.append(dfresultsGlobal)
    listdf_type.append(dfresultsGlobal[dfresultsGlobal["RealData [yes=1]"] == 1])
    listdf_type.append(dfresultsGlobal[dfresultsGlobal["RealData [yes=1]"] == 0])


    listNameTest = []
    listdf = []

    for i in range(listdf_type):

        typename = list_type[i]
        df = listdf_type[i]

        listNameTest.append(type)
        listdf.append(df)

        listNameTest.append(type + "_" + "Numerical-Numerical")
        listdf.append(dfresultsGlobal[(df["A type"] == "Numerical") & (df["B type"] == "Numerical")])
        listNameTest.append(type + "_" +"Categorical-Categorical")
        listdf.append(dfresultsGlobal[(df["A type"] == "Categorical") & (df["B type"] == "Categorical")])
        listNameTest.append(type + "_" + "Binary-Binary")
        listdf.append(dfresultsGlobal[(df["A type"] == "Binary") & (df["B type"] == "Binary")])
        listNameTest.append(type + "_" + "Numerical-Categorical")
        listdf.append(dfresultsGlobal[((df["A type"] == "Numerical") & (df["B type"] == "Categorical"))
                                    | ((df["B type"] == "Numerical") & (df["A type"] == "Categorical"))])
        listNameTest.append(type + "_" + "Numerical-Binary")
        listdf.append(dfresultsGlobal[((df["A type"] == "Numerical" & df["B type"] == "Binary"))
                                   | ( (df["B type"] == "Numerical") & (df["A type"] == "Binary"))])
        listNameTest.append(type + "_" + "Categorical-Binary")
        listdf.append(dfresultsGlobal[((df["A type"] == "Categorical") & (df["B type"] == "Binary"))
                                   | ( (df["B type"] == "Categorical") & (df["A type"] == "Binary"))])

        listNameTest.append(type + "_" + "num sample > 500")
        listdf.append(dfresultsGlobal[(df["sample num"] > 500)])

        listNameTest.append(type + "_" + "Numerical-Numerical" + "_" + "num sample > 500")
        listdf.append(dfresultsGlobal[(df["A type"] == "Numerical") & (df["B type"] == "Numerical") & (df["sample num"] > 500)])

        listNumSample = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 999999]

        for i in range(len(listNumSample)-1):
            listNameTest.append(type + "_" + "Num samples " + str(listNumSample[i]) + " - " +  str(listNumSample[i+1]))
            listdf.append(dfresultsGlobal[(df["sample num"] >= listNumSample[i]) & (df["sample num"] < listNumSample[i+1])])


    dfbenchmark = pd.DataFrame(columns=["Name test subset","Nb pairs", "ROC score avg", "ROC score A cause B", "ROC score B cause A"])

    for i in range(len(listdf)):
        nameTest = listNameTest[i]
        df = listdf[i]

        nbPairs = 0
        ROCscoreavg = 0
        ROCscoreAcauseB = 0
        ROCscoreBcauseA = 0

        if(df["Target"].count() > 0):
            ROCscoreAcauseB = scoreAcauseB(df["Final target"],df["Target"])
            ROCscoreBcauseA = scoreAcauseB(df["Final target"], df["Target"])

            ROCscoreavg = score(df["Final target"],df["Target"])
            nbPairs = df["Target"].count()

            print(nameTest)
            print(ROCscoreavg)
            print(nbPairs)

        newlignbenchmark = pd.DataFrame([[nameTest, nbPairs, ROCscoreavg, ROCscoreAcauseB,ROCscoreBcauseA ]], columns=["Name test subset","Nb pairs", "ROC score avg", "ROC score A cause B", "ROC score B cause A"])


    namebenchmark = "AllDataset"
    if(len(benchmarkname) == 1):
        namebenchmark = benchmarkname[0]

    dfbenchmark.to_csv("results_benchmark_" + namebenchmark + "_" + algo + '.csv', index=False, encoding='utf-8')


    listCausalThreshold = [0, 0.05,1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.50,0.55,0.60,0.65,0.7,0.75,0.8]

    dfthreshold = pd.DataFrame(columns=["name subset", "value threshold", "pr True A cause B", "pr Error A cause B", "pr True B cause A", "pr Error B cause A", "pr True causality", "pr Error causality", "pr true indep", "pr error indep" ])

    for i in range(listdf_type):

        typename = list_type[i]
        df = listdf_type[i]

        for threshold in listCausalThreshold:

            prTrueAcauseB = dfresultsGlobal[(dfresultsGlobal["Target"] >= threshold) & (dfresultsGlobal["Final Target"] == 1)].count() / (float)(dfresultsGlobal[(dfresultsGlobal["Final Target"] == 1)].count())
            prErrorAcauseB = dfresultsGlobal[(dfresultsGlobal["Target"] >= threshold) & (dfresultsGlobal["Final Target"] != 1)].count() / (float)(dfresultsGlobal[(dfresultsGlobal["Target"] >= threshold)])

            prTrueBcauseA = dfresultsGlobal[(dfresultsGlobal["Target"] <= -threshold) & (dfresultsGlobal["Final Target"] == -1)].count() / (float)(dfresultsGlobal[(dfresultsGlobal["Final Target"] == -1)].count())
            prErrorBcauseA = dfresultsGlobal[(dfresultsGlobal["Target"] <= -threshold) & (dfresultsGlobal["Final Target"] != -1)].count() / (float)(dfresultsGlobal[(dfresultsGlobal["Target"] <= -threshold)])

            prTruecausality = (dfresultsGlobal[(dfresultsGlobal["Target"] >= threshold) & (dfresultsGlobal["Final Target"] == 1)].count()
                             + dfresultsGlobal[(dfresultsGlobal["Target"] <= -threshold) & (dfresultsGlobal["Final Target"] == -1)].count())/ (float)(dfresultsGlobal[(dfresultsGlobal["Final Target"] == 1) | (dfresultsGlobal["Final Target"] == -1)].count())

            prErrorcausality = (dfresultsGlobal[(dfresultsGlobal["Target"] >= threshold) & (dfresultsGlobal["Final Target"] != 1)].count() + dfresultsGlobal[(dfresultsGlobal["Target"] <= -threshold) & (dfresultsGlobal["Final Target"] != -1)].count()) / (float)(dfresultsGlobal[(dfresultsGlobal["Target"] >= threshold) | (dfresultsGlobal["Target"] <= -threshold)])

            prTrueAindepB = dfresultsGlobal[(dfresultsGlobal["Target"] > -threshold) & (dfresultsGlobal["Target"] < threshold) & (dfresultsGlobal["Final Target"] == 0)].count() / (float)(dfresultsGlobal[(dfresultsGlobal["Final Target"] == 0)].count())
            prErrorAindepB = dfresultsGlobal[(dfresultsGlobal["Target"] > -threshold) & (dfresultsGlobal["Target"] < threshold) & (dfresultsGlobal["Final Target"] != 0)].count() /(float)(dfresultsGlobal[(dfresultsGlobal["Target"] > -threshold) & (dfresultsGlobal["Target"] < threshold)])

            newlignthreshold = pd.DataFrame([[typename, threshold, prTrueAcauseB, prErrorAcauseB, prTrueBcauseA,prErrorBcauseA,prTruecausality, prErrorcausality , prTrueAindepB, prErrorAindepB]], columns=["name subset","value threshold", "pr True A cause B", "pr Error A cause B", "pr True B cause A", "pr Error B cause A", "pr True causality", "pr Error causality", "pr true indep", "pr error indep" ])


    dfthreshold.to_csv("results_causal_threshold_" + namebenchmark + "_" + algo + '.csv', index=False, encoding='utf-8')


if __name__=="__main__":

    benchmarkpath = "output/benchmark/"

    targetpath = []
    privateinfopath = []
    benchmarkname = []

    # benchmarkname.append("SUP3")
    # targetpath.append("datacauseeffect/CEpairs/SUP3/CEdata_train_target.csv")
    # privateinfopath.append("datacauseeffect/CEpairs/SUP3/CEdata_train_privateinfo.csv")

    benchmarkname.append("validationset")
    targetpath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_target.csv")
    privateinfopath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_privateinfo.csv")

    # benchmarkname.append("validationset2")
    # targetpath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_target.csv")
    # privateinfopath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_privateinfo.csv")
    #
    # benchmarkname.append("validationset3")
    # targetpath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_target.csv")
    # privateinfopath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_privateinfo.csv")
    #
    # benchmarkname.append("validationset5")
    # targetpath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_target.csv")
    # privateinfopath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_privateinfo.csv")
    #
    # benchmarkname.append("validationset10")
    # targetpath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_target.csv")
    # privateinfopath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_privateinfo.csv")

    # listalgo = ["Fonolossa", "LopezKernel"]
    # listalgo = ["LopezKernel"]
    listalgo = ["Fonollosa"]

    for algo in listalgo:
        evalScore(benchmarkpath,benchmarkname, algo,targetpath,privateinfopath )




