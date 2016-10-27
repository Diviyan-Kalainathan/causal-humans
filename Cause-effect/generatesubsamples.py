
import numpy as np
import pandas as pd
from scipy import stats

inputfilespath = []
privateinfopath = []

# Benchmark test on Kaggle challenge data
# inputfilespath.append("datacauseeffect/CEpairs/SUP3/CEdata_train_pairs")
# privateinfopath.append("datacauseeffect/CEpairs/SUP3/CEdata_train_privateinfo")
#
# inputfilespath.append("datacauseeffect/CEpairs/SUP4/CEnovel_test_pairs")
# privateinfopath.append("datacauseeffect/CEpairs/SUP4/CEnovel_test_privateinfo")

inputfilespath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_pairs")
privateinfopath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_privateinfo")

ratiosubsamples = [2,3,5,10]


def transformData(x):

    tansformedData = ""
    for values in x:
        tansformedData += " " + str(float(values))

    return tansformedData


for idx in range(len(inputfilespath)):

    dfpair = pd.read_csv(inputfilespath[idx] + ".csv", index_col="SampleID")
    dfprivateinfo = pd.read_csv(privateinfopath[idx] + ".csv", index_col="SampleID")

    for ratio in ratiosubsamples:

        newdfpair = dfpair.copy()
        newdfprivateinfo = dfprivateinfo.copy()

        spearmanrcoefflist = []

        for i in range(0, newdfpair.shape[0]):

            if(i%100 == 0):
                print(i)

            x = np.array(newdfpair["A"].iloc[i].split(), dtype=np.float)[:, np.newaxis]
            y = np.array(newdfpair["B"].iloc[i].split(), dtype=np.float)[:, np.newaxis]

            n_samples = x.shape[0]
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            n_subset = min(n_samples, max(50,n_samples/ratio))


            x = x[indices[0:n_subset]]
            y = y[indices[0:n_subset]]


            xValuesParse = transformData(x)
            yValuesParse = transformData(y)

            newdfpair["A"].iat[i] = xValuesParse
            newdfpair["B"].iat[i] = yValuesParse

            newdfprivateinfo["sample num"].iat[i] = n_subset


        newdfpair.to_csv(inputfilespath[idx] + str(ratio) + ".csv", encoding="utf-8")
        newdfprivateinfo.to_csv(privateinfopath[idx] + str(ratio) + ".csv", encoding="utf-8")