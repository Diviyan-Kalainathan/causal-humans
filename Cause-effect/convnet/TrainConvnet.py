
from __future__ import print_function
import numpy as np

from sklearn.preprocessing import scale
from fastkde import fastKDE
from numpy import *

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

import pandas as pd

sys.path.insert(0, os.path.abspath(".."))  # For imports


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



def featurizePairs(pathinput,pairsfilename,targetfilename, publicinfofilename, filepairsvectorized, filetargetsvectorized, maxstd, size,  featurizingmethod = 0, ratio=1, doubletrainset=True, isTestSet = False, onlynumerical = False):


    print(pathinput)

    for i in range(0,len(pathinput)):

        dfpairs = pd.read_csv(pathinput[i] + pairsfilename[i] + ".csv", index_col="SampleID")
        dfpublicinfo = pd.read_csv(pathinput[i] + publicinfofilename[i] + ".csv", index_col="SampleID")

        if(isTestSet == False):
            dftarget = pd.read_csv(pathinput[i] + targetfilename[i] + ".csv", index_col="SampleID")


        if(i==0):
            dfpairsGlobal = dfpairs
            dfpublicinfoGlobal = dfpublicinfo

            if (isTestSet == False):
                dftargetGlobal = dftarget
        else:
            dfpairsGlobal = dfpairsGlobal.append(dfpairs)
            dfpublicinfoGlobal = dfpublicinfoGlobal.append(dfpublicinfo)

            if (isTestSet == False):
                dftargetGlobal = dftargetGlobal.append(dftarget)



    print("Total number of pairs to featurize : " + str(dfpairsGlobal.shape[0]))
    cpt = 0

    for k in range(0, int(dfpairsGlobal.shape[0]*ratio)):

        if(k%100==0):
            print(k)

        A = dfpairsGlobal['A'].iloc[k]
        B = dfpairsGlobal['B'].iloc[k]

        publicinfoA = dfpublicinfoGlobal['A type'].iloc[k]
        publicinfoB = dfpublicinfoGlobal['B type'].iloc[k]

        if (isTestSet == False):
            target = dftargetGlobal['Target'].iloc[k]

        if((publicinfoA == "Numerical" and publicinfoB == "Numerical") or onlynumerical == False):

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

                elif (featurizingmethod == 3):

                    pOfXGivenY, axes1 = fastKDE.conditional(x, y, numPoints=size+1, axisExpansionFactor=0.1)
                    pOfYGivenX, axes2 = fastKDE.conditional(y, x, numPoints=size+1, axisExpansionFactor=0.1)


                    # pXY = np.dstack((pOfYGivenX, np.transpose(pOfXGivenY)))
                    pXY = np.dstack((pOfYGivenX, pOfXGivenY))

                    pXY = delete(pXY, s_[0], axis=0)
                    pXY = delete(pXY, s_[0], axis=1)

                    pYX = np.dstack((pOfXGivenY, pOfYGivenX))
                    pYX = delete(pYX, s_[0], axis=0)
                    pYX = delete(pYX, s_[0], axis=1)

                arrayXY = np.ravel(pXY)
                # arrayXY = pXY

                if(cpt==0):
                    vectorizedPairs = arrayXY
                else:
                    vectorizedPairs = np.vstack((vectorizedPairs, arrayXY))

                if(doubletrainset == True):

                    if (featurizingmethod == 3):
                        arrayYX = np.ravel(pYX)

                    else:
                        arrayYX = np.ravel(np.transpose(pXY))


                    vectorizedPairs = np.vstack((vectorizedPairs, arrayYX))

                if(isTestSet == False):
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


    np.savetxt(filepairsvectorized, vectorizedPairs)

    if (isTestSet == False):
        np.savetxt(filetargetsvectorized, vectorizedTarget)
        return vectorizedPairs,vectorizedTarget
    else:
        return vectorizedPairs




# featurize parameters
alreadyFeaturized = False
size = 48                   # Image size x size
maxstd = 3                  # max std for outliers removal
ratio = 1                   # ratio training set we keep
doubletrainset = True      # duplicate training set
featurizingmethod = 0       # 0 for histo, 1 for kernel
onlynumericalpairs = False



modelpath = "../lib/causal_deep/model.h5"

nameSimu = "Test"

# train dataset from Kaggle
trainpath = []
trainfilename = []
traintargetfilename = []
trainpublicinfofilename = []

trainpath.append("../datacauseeffect/CEpairs/CEdata/")
trainfilename.append("CEfinal_train_pairs")
traintargetfilename.append("CEfinal_train_target")
trainpublicinfofilename.append("CEfinal_train_publicinfo")

# SUP1 train dataset from Kaggle
trainpath.append("../datacauseeffect/CEpairs/SUP1/")
trainfilename.append("CEdata_train_pairs")
traintargetfilename.append("CEdata_train_target")
trainpublicinfofilename.append("CEdata_train_publicinfo")

# SUP2 train dataset from Kaggle
trainpath.append("../datacauseeffect/CEpairs/SUP2/")
trainfilename.append("CEdata_train_pairs")
traintargetfilename.append("CEdata_train_target")
trainpublicinfofilename.append("CEdata_train_publicinfo")

# SUP3 train dataset from Kaggle
trainpath.append("../datacauseeffect/CEpairs/SUP3/")
trainfilename.append("CEdata_train_pairs")
traintargetfilename.append("CEdata_train_target")
trainpublicinfofilename.append("CEdata_train_publicinfo")

# SUP4 train dataset from Kaggle
trainpath.append("../datacauseeffect/CEpairs/SUP4/")
trainfilename.append("CEnovel_test_pairs")
traintargetfilename.append("CEnovel_test_target")
trainpublicinfofilename.append("CEnovel_test_publicinfo")


# Validation path
validpath = []
validfilename = []
validtargetfilename = []
validpublicinfofilename = []

validpath.append("../datacauseeffect/CEpairs/CEdata/")
validfilename.append("CEfinal_valid_pairs")
validtargetfilename.append("CEfinal_valid_target")
validpublicinfofilename.append("CEfinal_valid_publicinfo")



filetrainpairsvectorized = "../output/featurizedPairs/vectorized"+ nameSimu + "_maxstd3_size" + str(size) + "_method" + str(featurizingmethod) + "CEfinal_train_pairs.csv"
filetraintargetsvectorized = "../output/featurizedPairs/vectorized"+ nameSimu + "_maxstd3_size" + str(size) + "_method" + str(featurizingmethod) + "CEfinal_train_target.csv"
filevalidpairsvectorized = "../output/featurizedPairs/vectorized"+ nameSimu + "_maxstd3_size" + str(size) + "_method" + str(featurizingmethod) + "CEfinal_valid_pairs.csv"
filevalidtargetsvectorized = "../output/featurizedPairs/vectorized"+ nameSimu + "_maxstd3_size" + str(size) + "_method" + str(featurizingmethod) + "CEfinal_valid_target.csv"




print("Build network")

img_rows, img_cols = size, size

if(featurizingmethod == 3):
    input_shape = (img_rows, img_cols, 2)
else :
    input_shape = (img_rows, img_cols, 1)

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size

kernel_size = (8, 8)

# kernel_size = (8,8)
nb_classes = 3
# sizeHiddenLayer = 256

batch_size_value = 1

model = Sequential()

model.add(ZeroPadding2D((1, 1), batch_input_shape = (batch_size_value,img_rows, img_cols, 1), name='inputlayer'))
first_layer = model.layers[-1]
# this is a placeholder tensor that will contain our generated images
input_img = first_layer.input

droupoutconvlayers = 0.18

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='same',
                        input_shape=input_shape, name='conv1'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same', name='conv2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(droupoutconvlayers))



model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same', name='conv3'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same', name='conv4'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(droupoutconvlayers))

# model.add(Convolution2D(nb_filters, newsize, newsize, border_mode='same'))
# model.add(Activation('relu'))
# model.add(Convolution2D(nb_filters, newsize, newsize, border_mode='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(droupoutconvlayers))


droupoutdenslayers = 0.25


model.add(Flatten())
model.add(Dense(1024,  name='dense1'))
model.add(Activation('relu'))
model.add(Dropout(droupoutdenslayers))

model.add(Dense(256,  name='dense2'))
model.add(Activation('relu'))
model.add(Dropout(droupoutdenslayers))

model.add(Dense(nb_classes,  name='ouputlayer'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])



if(alreadyFeaturized == True):
    print("Load data")
    X_valid = np.loadtxt(filevalidpairsvectorized)
    Y_valid = np.loadtxt(filevalidtargetsvectorized)
    print("Valid data loaded")
    X_train = np.loadtxt(filetrainpairsvectorized)
    Y_train = np.loadtxt(filetraintargetsvectorized)
    print("Train data loaded")


else :
    print("Featurize validation data")
    X_valid,Y_valid = featurizePairs(validpath,validfilename,validtargetfilename, validpublicinfofilename, filevalidpairsvectorized, filevalidtargetsvectorized, maxstd, size, featurizingmethod, ratio, doubletrainset = False, onlynumerical = onlynumericalpairs )
    print("Featurize train data")
    X_train,Y_train = featurizePairs(trainpath,trainfilename,traintargetfilename, trainpublicinfofilename, filetrainpairsvectorized,filetraintargetsvectorized, maxstd, size, featurizingmethod,ratio, doubletrainset, onlynumerical = onlynumericalpairs)



print("Train data loaded")

if(featurizingmethod == 3):
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 2)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 2)
else :
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols,1)


X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')



print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'test samples')

print("Start learning")
_num_examples = X_train.shape[0]


print("nb examples for training : " + str(_num_examples))
_index_in_epoch = 0
_epochs_completed = 0

model.fit(X_train, Y_train,
          batch_size=batch_size_value,
          nb_epoch=2,
          show_accuracy=True,
          verbose=2,
          validation_data=(X_valid, Y_valid))



model.save(modelpath)








