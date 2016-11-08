from __future__ import print_function
import numpy as np
from sklearn.preprocessing import scale
from fastkde import fastKDE
from numpy import *
from keras.models import load_model

import pandas as pd




def getHisto(x, y, size, maxstd):

    intx = np.round(x * (size) / maxstd / 2)
    inty = np.round(y * (size) / maxstd / 2)

    pXY = np.zeros((size, size))

    for i in range(len(x)):
        coordx = int(min(intx[i], size / 2 - 1) + size / 2)
        coordy = int(min(inty[i], size / 2 - 1) + size / 2)

        pXY[coordx, coordy] += 1

    maxPXY = np.amax(pXY)

    pXY = pXY / maxPXY

    return pXY



def featurizePairs(pathinput, maxstd, size):


    dfpairs = pd.read_csv(pathinput , index_col="SampleID")


    print("Total number of pairs to featurize : " + str(dfpairs.shape[0]))
    cpt = 0

    for k in range(0, int(dfpairs.shape[0])):

        if(k%100==0):
            print(k)

        A = dfpairs['A'].iloc[k]
        B = dfpairs['B'].iloc[k]

        x = scale(np.array(A.split(), dtype=np.float))
        y = scale(np.array(B.split(), dtype=np.float))

        mask = (x > -maxstd) & (x < maxstd) & ( y > -maxstd) & ( y < maxstd)
        x = x[mask]
        y = y[mask]


        pXY = getHisto(x, y, size, maxstd)

        arrayXY = np.ravel(pXY)

        if(cpt==0):
            vectorizedPairs = arrayXY
        else:
            vectorizedPairs = np.vstack((vectorizedPairs, arrayXY))

        cpt = cpt + 1

    return vectorizedPairs




def predict(modelpath, inputpath, outputpath, size = 48, maxstd = 3 ):


    img_rows, img_cols = size, size

    model = load_model(modelpath)

    X_te = featurizePairs(inputpath,size, maxstd)


    X_te = X_te.reshape(X_te.shape[0], img_rows, img_cols,1)
    X_te = X_te.astype('float32')

    resultproba = model.predict_proba(X_te, batch_size=32, verbose=1)

    scorepairs = (resultproba[:,2] - resultproba[:,0])*(1-resultproba[:,1])

    df = pd.read_csv(inputpath, index_col="SampleID")

    Results = pd.DataFrame(index=df.index)
    Results['Target'] = scorepairs

    Results.to_csv(outputpath, sep=';', encoding='utf-8')


