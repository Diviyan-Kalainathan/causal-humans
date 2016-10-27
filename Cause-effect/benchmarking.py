

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


def score(y,p):

  try:
    ROCscore = (roc_auc_score(y==1,p)+roc_auc_score(y==-1,-p))/2
  except(ValueError):
    ROCscore = np.nan

  return ROCscore

def scoreAcauseB(y,p):
  try:
      ROCscore =  roc_auc_score(y==1,p)
  except(ValueError):
      ROCscore = np.nan

  return ROCscore

def scoreBcauseA(y,p):

  try:
      ROCscore =  roc_auc_score(y==-1,-p)
  except(ValueError):
      ROCscore = np.nan

  return ROCscore


def evalScore(benchmarkpath,benchmarkname, algo,targetpath,privateinfopath, outputROCscorepath, outputcausalitythresholdpath ):

    # try:
    dfresultsGlobal = pd.read_csv(benchmarkpath + algo + "_" + benchmarkname[0] + ".csv", sep =',|;',index_col="SampleID")
    # except:
    #     dfresultsGlobal = pd.read_csv(benchmarkpath + algo + "_" + benchmarkname[0] + ".csv", index_col="SampleID", sep=",")

    dftargetGlobal = pd.read_csv(targetpath[0],index_col="SampleID")
    dfprivateinfoGlobal = pd.read_csv(privateinfopath[0],index_col="SampleID")



    for i in range(1,len(benchmarkname)):

        dfresults = pd.read_csv(benchmarkpath + algo + "_" + benchmarkname[i] + ".csv",sep =',|;',index_col="SampleID")


        dftarget = pd.read_csv(targetpath[i],index_col="SampleID")
        dfprivateinfo = pd.read_csv(privateinfopath[i],index_col="SampleID")

        dfresultsGlobal = dfresultsGlobal.append(dfresults)
        dftargetGlobal = dftargetGlobal.append(dftarget)
        dfprivateinfoGlobal = dfprivateinfoGlobal.append(dfprivateinfo)



    dfresultsGlobal["Final target"] = dftargetGlobal["Target"]
    dfresultsGlobal["Source"] = dfprivateinfoGlobal["Source"]
    dfresultsGlobal["A type"] = dfprivateinfoGlobal["A type"]
    dfresultsGlobal["B type"] = dfprivateinfoGlobal["B type"]
    dfresultsGlobal["sample num"] = dfprivateinfoGlobal["sample num"]
    dfresultsGlobal["RealData [yes=1]"] = dfprivateinfoGlobal["RealData [yes=1]"]
    dfresultsGlobal["spearmancoeff"] = dfprivateinfoGlobal["spearmancoeff"]


    listdf_type = []
    list_type = ["All data", "Real data","Artificial data" ]

    listdf_type.append(dfresultsGlobal)
    listdf_type.append(dfresultsGlobal[dfresultsGlobal["RealData [yes=1]"] == 1])
    listdf_type.append(dfresultsGlobal[dfresultsGlobal["RealData [yes=1]"] == 0])


    listNameTest = []
    listdf = []

    for i in range(len(listdf_type)):

        type = list_type[i]
        df = listdf_type[i]

        listNameTest.append(type)
        listdf.append(df)


        listNameTest.append(type + "_" + "Numerical-Numerical")
        listdf.append(df[(df["A type"] == "Numerical") & (df["B type"] == "Numerical")])
        listNameTest.append(type + "_" +"Categorical-Categorical")
        listdf.append(df[(df["A type"] == "Categorical") & (df["B type"] == "Categorical")])
        listNameTest.append(type + "_" + "Binary-Binary")
        listdf.append(df[(df["A type"] == "Binary") & (df["B type"] == "Binary")])
        listNameTest.append(type + "_" + "Numerical-Categorical")
        listdf.append(df[((df["A type"] == "Numerical") & (df["B type"] == "Categorical")) | ((df["B type"] == "Numerical") & (df["A type"] == "Categorical"))])
        listNameTest.append(type + "_" + "Numerical-Binary")
        listdf.append(df[((df["A type"] == "Numerical") & (df["B type"] == "Binary")) | ( (df["B type"] == "Numerical") & (df["A type"] == "Binary"))])
        listNameTest.append(type + "_" + "Categorical-Binary")
        listdf.append(df[((df["A type"] == "Categorical") & (df["B type"] == "Binary")) | ( (df["B type"] == "Categorical") & (df["A type"] == "Binary"))])

        listNameTest.append(type + "_" + "num sample > 500")
        listdf.append(df[(df["sample num"] > 500)])

        listNameTest.append(type + "_" + "Numerical-Numerical" + "_" + "num sample > 500")
        listdf.append(df[(df["A type"] == "Numerical") & (df["B type"] == "Numerical") & (df["sample num"] > 500)])

        listNameTest.append(type + "_" + "spearmancoeff > 0.2")
        listdf.append(df[(df["spearmancoeff"] > 0.2) | (df["spearmancoeff"] < -0.2)])

        listNameTest.append(type + "_" + "spearmancoeff > 0.2" + "_" + "num sample > 500")
        listdf.append(df[((df["spearmancoeff"] > 0.2) | (df["spearmancoeff"] < -0.2)) & (df["sample num"] > 500)])

        listNameTest.append(type + "_" + "spearmancoeff > 0.2" + "_" + "num sample > 500" + "_" + "Numerical-Numerical")
        listdf.append(df[((df["spearmancoeff"] > 0.2) | (df["spearmancoeff"] < -0.2)) & (df["sample num"] > 500) & (df["A type"] == "Numerical") & (df["B type"] == "Numerical")])

        listNumSample = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 999999]

        for i in range(len(listNumSample)-1):
            listNameTest.append(type + "_" + "Num samples " + str(listNumSample[i]) + " - " +  str(listNumSample[i+1]))
            listdf.append(df[(df["sample num"] >= listNumSample[i]) & (df["sample num"] < listNumSample[i+1])])


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
            ROCscoreBcauseA = scoreBcauseA(df["Final target"], df["Target"])

            ROCscoreavg = score(df["Final target"],df["Target"])
            nbPairs = df["Target"].count()

            print(nameTest)
            print(ROCscoreavg)
            print(nbPairs)

        newlignbenchmark = pd.DataFrame([[nameTest, nbPairs, ROCscoreavg, ROCscoreAcauseB,ROCscoreBcauseA ]], columns=["Name test subset","Nb pairs", "ROC score avg", "ROC score A cause B", "ROC score B cause A"])
        dfbenchmark = dfbenchmark.append(newlignbenchmark)

    namebenchmark = "AllDataset"
    if(len(benchmarkname) == 1):
        namebenchmark = benchmarkname[0]

    dfbenchmark.to_csv(outputROCscorepath + "ROCscore_" + namebenchmark + "_" + algo + '.csv', index=False, encoding='utf-8')


    listCausalThreshold = np.linspace(-1, 1, num=41)

    dfthreshold = pd.DataFrame(columns=["name subset", "value threshold", "accuracy", "pr False A cause B","pr True A cause B", "pr Error A cause B parmis detectes", "pr False B cause A","pr True B cause A", "pr Error B cause A parmis detectes", "pr False causality", "pr True causality", "pr Error causality", "pr wrong sens causality", "pr fasle indep", "pr true indep", "pr error indep" ])

    for i in range(len(listdf_type)):

        typename = list_type[i]
        df = listdf_type[i]
        if(df.shape[0]>0):
            for threshold in listCausalThreshold:

                accuracy = (df[(df["Target"] >= threshold) & (df["Final target"] == 1)].shape[0]
                                 + df[(df["Target"] <= -threshold) & (df["Final target"] == -1)].shape[0] + df[(df["Target"] > -threshold) & (df["Target"] < threshold) & (df["Final target"] == 0)].shape[0])/ (float)(df.shape[0])

                prTrueAcauseB = df[(df["Target"] >= threshold) & (df["Final target"] == 1)].shape[0] / (float)(df[(df["Final target"] == 1)].shape[0])
                prFalseAcauseB = df[(df["Target"] >= threshold) & (df["Final target"] != 1)].shape[0] / (float)(df[(df["Final target"] != 1)].shape[0])


                if(df[(df["Target"] >= threshold)].shape[0] > 0):
                    prErrorAcauseB = df[(df["Target"] >= threshold) & (df["Final target"] != 1)].shape[0] / (float)(df[(df["Target"] >= threshold)].shape[0])
                else :
                    prErrorAcauseB = 0

                prTrueBcauseA = df[(df["Target"] <= -threshold) & (df["Final target"] == -1)].shape[0] / (float)(df[(df["Final target"] == -1)].shape[0])
                prFalseBcauseA = df[(df["Target"] <= -threshold) & (df["Final target"] != -1)].shape[0] / (float)(df[(df["Final target"] != -1)].shape[0])

                if(df[(df["Target"] <= -threshold)].shape[0] > 0):
                    prErrorBcauseA = df[(df["Target"] <= -threshold) & (df["Final target"] != -1)].shape[0] / (float)(df[(df["Target"] <= -threshold)].shape[0])
                else:
                    prErrorBcauseA = 0

                prTruecausality = (df[(df["Target"] >= threshold) & (df["Final target"] == 1)].shape[0]
                                 + df[(df["Target"] <= -threshold) & (df["Final target"] == -1)].shape[0])/ (float)(df[(df["Final target"] == 1) | (df["Final target"] == -1)].shape[0])

                prFalsecausality = (df[(df["Target"] >= threshold) & (df["Final target"] != 1)].shape[0] + df[(df["Target"] <= -threshold) & (df["Final target"] != -1)].shape[0]) / (float)(
                    df[(df["Final target"] != 1) | (df["Final target"] != -1)].shape[0])


                if(df[(df["Target"] >= threshold) | (df["Target"] <= -threshold)].shape[0] > 0):
                    prErrorcausality = (df[(df["Target"] >= threshold) & (df["Final target"] != 1)].shape[0] + df[(df["Target"] <= -threshold) & (df["Final target"] != -1)].shape[0]) / (float)(df[(df["Target"] >= threshold) | (df["Target"] <= -threshold)].shape[0])
                    prErrorSensCausality = (df[(df["Target"] >= threshold) & (df["Final target"] == -1)].shape[0] +
                                            df[(df["Target"] <= -threshold) & (df["Final target"] == 1)].shape[0]) / (
                                           float)(
                        df[(df["Target"] >= threshold) | (df["Target"] <= -threshold)].shape[0])

                else:
                    prErrorSensCausality = 0
                    prErrorcausality = 0



                prTrueAindepB = df[(df["Target"] > -threshold) & (df["Target"] < threshold) & (df["Final target"] == 0)].shape[0] / (float)(df[(df["Final target"] == 0)].shape[0])
                prFalseAindepB = df[(df["Target"] > -threshold) & (df["Target"] < threshold) & (df["Final target"] != 0)].shape[0] / (float)(df[(df["Final target"] != 0)].shape[0])

                if(df[(df["Target"] > -threshold) & (df["Target"] < threshold)].shape[0] > 0):
                    prErrorAindepB = df[(df["Target"] > -threshold) & (df["Target"] < threshold) & (df["Final target"] != 0)].shape[0] /(float)(df[(df["Target"] > -threshold) & (df["Target"] < threshold)].shape[0])
                else:
                    prErrorAindepB = 0

                newlignthreshold = pd.DataFrame([[typename, threshold,accuracy, prFalseAcauseB,prTrueAcauseB, prErrorAcauseB, prFalseBcauseA, prTrueBcauseA, prErrorBcauseA,prFalsecausality,prTruecausality, prErrorcausality ,prErrorSensCausality, prFalseAindepB,prTrueAindepB, prErrorAindepB]], columns=["name subset", "value threshold","accuracy", "pr False A cause B","pr True A cause B", "pr Error A cause B parmis detectes", "pr False B cause A","pr True B cause A", "pr Error B cause A parmis detectes", "pr False causality", "pr True causality", "pr Error causality", "pr wrong sens causality", "pr fasle indep", "pr true indep", "pr error indep" ])
                dfthreshold = dfthreshold.append(newlignthreshold)

    dfthreshold.to_csv(outputcausalitythresholdpath + "results_causality_threshold_" + namebenchmark + "_" + algo + '.csv', index=False, encoding='utf-8')


if __name__=="__main__":

    resultpredictpath = "output/resultpredict/"
    outputROCscorepath = "output/benchmark/ROCscore/"
    outputcausalitythresholdpath = "output/benchmark/causalitythreshold/"

    targetpath = []
    privateinfopath = []
    testname = []

    # testname.append("SUP3")
    # targetpath.append("datacauseeffect/CEpairs/SUP3/CEdata_train_target.csv")
    # privateinfopath.append("datacauseeffect/CEpairs/SUP3/CEdata_train_privateinfo1.csv")

    # testname.append("SUP4")
    # targetpath.append("datacauseeffect/CEpairs/SUP4/CEnovel_test_target.csv")
    # privateinfopath.append("datacauseeffect/CEpairs/SUP4/CEnovel_test_privateinfo1.csv")

    testname.append("validationset")
    targetpath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_target.csv")
    privateinfopath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_privateinfo1.csv")

    # testname.append("validationset2")
    # targetpath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_target.csv")
    # privateinfopath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_privateinfo1.csv")
    #
    # testname.append("validationset3")
    # targetpath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_target.csv")
    # privateinfopath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_privateinfo1.csv")
    #
    # testname.append("validationset5")
    # targetpath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_target.csv")
    # privateinfopath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_privateinfo1.csv")
    #
    # testname.append("validationset10")
    # targetpath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_target.csv")
    # privateinfopath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_privateinfo1.csv")

    # listalgo = ["LopezCodalab"]
    listalgo = ["LopezKernel", "LopezCodalab", "Fonollosa"]
    # listalgo = ["LopezKernel"]

    for algo in listalgo:
        evalScore(resultpredictpath,testname, algo,targetpath,privateinfopath, outputROCscorepath, outputcausalitythresholdpath )




