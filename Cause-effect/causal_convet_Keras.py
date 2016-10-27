
from __future__ import print_function
import numpy as np
from keras.regularizers import l2
from sklearn.preprocessing import scale
from fastkde import fastKDE
from numpy import *

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LocallyConnected2D
import keras.optimizers
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



def featurizePairs(pathinput,pairsfilename,targetfilename, publicinfofilename, pathoutput, maxstd, size, featurizingmethod = 0, ratio=1, doubletrainset=True):


    print(pathinput)

    for i in range(0,len(pathinput)):

        dfpairs = pd.read_csv(pathinput[i] + pairsfilename[i] + ".csv", index_col="SampleID")

        dftarget = pd.read_csv(pathinput[i] + targetfilename[i] + ".csv", index_col="SampleID")
        dfpublicinfo = pd.read_csv(pathinput[i] + publicinfofilename[i] + ".csv", index_col="SampleID")

        if(i==0):
            dfpairsGlobal = dfpairs
            dftargetGlobal = dftarget
            dfpublicinfoGlobal = dfpublicinfo
        else:
            dfpairsGlobal = dfpairsGlobal.append(dfpairs)
            dftargetGlobal = dftargetGlobal.append(dftarget)
            dfpublicinfoGlobal = dfpublicinfoGlobal.append(dfpublicinfo)


    #
    # f = open(path + filepairs );
    # pairs = f.readlines();
    # pairs.pop(0)
    # f.close();

    # y_te = np.genfromtxt(path + filetargets, delimiter=",")

    print(dfpairsGlobal.shape[0])

    cpt = 0

    for k in range(0, int(dfpairsGlobal.shape[0]*ratio)):

        if(k%100==0):
            print(k)

        A = dfpairsGlobal['A'].iloc[k]
        B = dfpairsGlobal['B'].iloc[k]
        target = dftargetGlobal['Target'].iloc[k]
        publicinfoA = dfpublicinfoGlobal['A type'].iloc[k]
        publicinfoB = dfpublicinfoGlobal['B type'].iloc[k]




        if(publicinfoA == "Numerical" and publicinfoB == "Numerical"):

            x = scale(np.array(A.split(), dtype=np.float))
            y = scale(np.array(B.split(), dtype=np.float))


            mask = (x > -maxstd) & (x < maxstd) & ( y > -maxstd) & ( y < maxstd)
            x = x[mask]
            y = y[mask]

            try:
                if(featurizingmethod == 0):
                    pXY = getHisto(x, y, size, maxstd)

                elif(featurizingmethod == 1):

                    pXY, axes = fastKDE.pdf(x, y, numPoints=size+1, axisExpansionFactor = 0.1)

                    pXY = delete(pXY, s_[0], axis=0)
                    pXY = delete(pXY, s_[0], axis=1)

                elif (featurizingmethod == 2):

                    pOfXGivenY, axes = fastKDE.conditional(x, y, numPoints=size+1, axisExpansionFactor=0.1)
                    pOfYGivenX, axes = fastKDE.conditional(y, x, numPoints=size+1, axisExpansionFactor=0.1)

                    pOfXGivenY = delete(pOfXGivenY, s_[0], axis=0)
                    pOfXGivenY = delete(pOfXGivenY, s_[0], axis=1)

                    pOfYGivenX = delete(pOfYGivenX, s_[0], axis=0)
                    pOfYGivenX = delete(pOfYGivenX, s_[0], axis=1)

                    pXY = np.hstack((pOfYGivenX, pOfXGivenY))


                arrayXY = np.ravel(pXY)

                if(cpt==0):
                    vectorizedPairs = arrayXY
                else:
                    vectorizedPairs = np.vstack((vectorizedPairs, arrayXY))

                if(doubletrainset == True):
                    arrayYX = np.ravel(np.transpose(pXY))
                    vectorizedPairs = np.vstack((vectorizedPairs, arrayYX))

                if target == -1:
                    arrayTargetXY = np.array([1,0,0])
                    arrayTargetYX = np.array([0,0,1])

                elif target == 0:
                    arrayTargetXY = np.array([0,1,0])
                    arrayTargetYX = np.array([0,1,0])

                elif target == 1:
                    arrayTargetXY = np.array([0,0,1])
                    arrayTargetYX = np.array([1,0,0])

                if(cpt==0):
                    vectorizedTarget = arrayTargetXY
                else:
                    vectorizedTarget = np.vstack((vectorizedTarget, arrayTargetXY))

                if (doubletrainset == True):
                    vectorizedTarget = np.vstack((vectorizedTarget, arrayTargetYX))

                cpt = cpt + 1

            except ValueError:
                print("pbkde nbpoints pairs " + str(k))


    np.savetxt(pathoutput + "vectorized" + "_maxstd" + str(maxstd) + "_size" + str(size) + "_method" + str(featurizingmethod) + pairsfilename[0], vectorizedPairs)
    np.savetxt(pathoutput + "vectorized" + "_maxstd" + str(maxstd) + "_size" + str(size) + "_method" + str(featurizingmethod) + targetfilename[0], vectorizedTarget)

    return vectorizedPairs,vectorizedTarget



def getNextBatch(batch_size):

    global _index_in_epoch
    global _num_examples
    global _epochs_completed
    global X_train
    global Y_train

    start = _index_in_epoch
    _index_in_epoch += batch_size

    newEpoch = False
    if _index_in_epoch > _num_examples:
          # Finished epoch
      print("shuffle")
      _epochs_completed += 1
      print("_epochs_completed " + str(_epochs_completed))
      newEpoch = True
      # Shuffle the data
      perm = np.arange(_num_examples)
      np.random.shuffle(perm)
      X_train = X_train[perm]
      Y_train = Y_train[perm]
      # Start next epoch
      start = 0
      _index_in_epoch = batch_size
      assert batch_size <= _num_examples

    end = _index_in_epoch
    return X_train[start:end], Y_train[start:end], newEpoch




# featurize parameters
alreadyFeaturized = True
size = 32              # Image size x size
maxstd = 2              # max std for outliers removal
ratio = 1             # ratio training set we keep
doubletrainset = True   # duplicate training set
featurizingmethod = 0   # 0 for histo, 1 for kernel


# network parameters

# input image dimensions
img_rows, img_cols = size, size
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (8, 8)
nb_classes = 3
sizeHiddenLayer = 1024


# training parameters
batch_size = 50
nb_epoch = 200
# regularizeParam = 0.01
regularizeParam = 0.0005
dropoutRate = 0.5
cptEarlyStop = 100
ratiosplitTrainset = 1

optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
# optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


pathoutput = "output/"
# filetrainpairs = "train_pairs.csv"
# filetraintargets = "train_target.csv"
# filevalidpairs = "CEfinal_valid_pairs.csv"
# filevalidtargets = "CEfinal_valid_target.csv"

# train dataset from Kaggle
trainpath = []
trainfilename = []
traintargetfilename = []
trainpublicinfofilename = []

trainpath.append("datacauseeffect/CEpairs/CEdata/")
trainfilename.append("CEfinal_train_pairs")
traintargetfilename.append("CEfinal_train_target")
trainpublicinfofilename.append("CEfinal_train_publicinfo")

# SUP1 train dataset from Kaggle
trainpath.append("datacauseeffect/CEpairs/SUP1/")
trainfilename.append("CEdata_train_pairs")
traintargetfilename.append("CEdata_train_target")
trainpublicinfofilename.append("CEdata_train_publicinfo")

# SUP2 train dataset from Kaggle
trainpath.append("datacauseeffect/CEpairs/SUP2/")
trainfilename.append("CEdata_train_pairs")
traintargetfilename.append("CEdata_train_target")
trainpublicinfofilename.append("CEdata_train_publicinfo")

# Validation path
validpath = []
validfilename = []
validtargetfilename = []
validpublicinfofilename = []

validpath.append("datacauseeffect/CEpairs/CEdata/")
validfilename.append("CEfinal_valid_pairs")
validtargetfilename.append("CEfinal_valid_target")
validpublicinfofilename.append("CEfinal_valid_publicinfo")



filetrainpairsvectorized = "output/vectorized_maxstd2_size32_method0CEfinal_train_pairs.csv"
filetraintargetsvectorized = "output/vectorized_maxstd2_size32_method0CEfinal_train_target.csv"
filevalidpairsvectorized = "output/vectorized_maxstd2_size32_method0CEfinal_valid_pairs.csv"
filevalidtargetsvectorized = "output/vectorized_maxstd2_size32_method0CEfinal_valid_target.csv"

# filetrainpairsvectorized = "data/vectorizedtrain_numpoints" + str(size) + "_maxstd2.csv"
# filetraintargetsvectorized = "data/vectorizedtraintarget.csv"
# filevalidpairsvectorized = "data/vectorizedvalid_numpoints" + str(size) + "_maxstd2.csv"
# filevalidtargetsvectorized = "data/vectorizedvalidtarget.csv"


print("Build network")

input_shape = (img_rows,img_cols,1)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape, W_regularizer=l2(regularizeParam)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], W_regularizer=l2(regularizeParam)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(dropoutRate))

model.add(Flatten())
model.add(Dense(sizeHiddenLayer, W_regularizer=l2(regularizeParam)))
model.add(Activation('relu'))
model.add(Dropout(dropoutRate))
model.add(Dense(sizeHiddenLayer, W_regularizer=l2(regularizeParam)))
model.add(Activation('relu'))
model.add(Dropout(dropoutRate))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])



if(alreadyFeaturized == True):
    print("Load data")
    X_test = np.loadtxt(filevalidpairsvectorized)
    Y_test = np.loadtxt(filevalidtargetsvectorized)
    print("Valid data loaded")
    X_train = np.loadtxt(filetrainpairsvectorized)
    Y_train = np.loadtxt(filetraintargetsvectorized)
    print("Train data loaded")
else :
    print("Featurize validation data")
    X_test,Y_test = featurizePairs(validpath,validfilename,validtargetfilename, validpublicinfofilename, pathoutput, maxstd, size, featurizingmethod, ratio, doubletrainset = False)
    print("Featurize train data")
    X_train,Y_train = featurizePairs(trainpath,trainfilename,traintargetfilename, trainpublicinfofilename, pathoutput, maxstd, size, featurizingmethod,ratio, doubletrainset)

print("Train data loaded")

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print("Start learning")
_num_examples = X_train.shape[0]
globalBatchsize = _num_examples * ratiosplitTrainset

print("nb examples for training : " + str(_num_examples))
_index_in_epoch = 0
_epochs_completed = 0


maxaccurary = 0
cptAmelio = 0
i = 0

# while i < nb_epoch and cptAmelio < cptEarlyStop:
#
#   batch = getNextBatch(globalBatchsize)
#
#   model.fit(batch[0], batch[1],batch_size=batch_size, nb_epoch=1, verbose = 1 )
#
#   accuracy = model.evaluate(X_test, Y_test, verbose=0)[1]
#   print("test accuracy validation " + str(accuracy))
#
#   cptAmelio = cptAmelio + 1
#
#   if(accuracy > maxaccurary):
#       maxaccurary = accuracy
#       cptAmelio = 0
#
#   print("max accuracy " + str(maxaccurary))
#   i = i + 1



model.fit(X_train, Y_train,batch_size=batch_size, nb_epoch=nb_epoch, verbose = 1 )


score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])