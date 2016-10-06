
import numpy as np
import pandas as pd

inputfilespath = []
privateinfopath = []

# Benchmark test on Kaggle challenge data
# SUP3 Test coming from Tuebingen pairs
inputfilespath.append("datacauseeffect/CEpairs/SUP3/CEdata_train_pairs.csv")
privateinfopath.append("datacauseeffect/CEpairs/SUP3/CEdata_train_privateinfo.csv")

ratiosubsamples = [2,3,5,10]



def transformData(x):

    tansformedData = ""
    for values in x:
        tansformedData += " " + str(float(values))

    return tansformedData


for idx in len(inputfilespath):

    dfpair = pd.read_csv(inputfilespath)
    dfprivateinfo = pd.read_csv(privateinfopath)


    for ratio in ratiosubsamples:

        newdfpair = pd.copy(dfpair)
        newdfprivateinfo = pd.copy(dfprivateinfo)

        for i in range(0, df.shape[0]):

            r = row.split(",", 2)

            x = np.array(r[1].split(), dtype=np.float)[:, np.newaxis]
            y = np.array(r[2].split(), dtype=np.float)[:, np.newaxis]

            n_samples = x.shape[0]
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            n_subset = n_samples/ratio
            print(n_subset)
            x = x[indices[0:n_subset]]
            y = y[indices[0:n_subset]]

            xValuesParse = transformData(x)
            yValuesParse = transformData(y)

            newLignPairs = pd.DataFrame([[nameDistrib, xValuesParse, yValuesParse]], columns=["SampleID", "A", "B"])

            newLignPublicInfo = pd.DataFrame([[nameDistrib, "Numerical", "Numerical"]],columns=["SampleID", "A type", "B type"])

