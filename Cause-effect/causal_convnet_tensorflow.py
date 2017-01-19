
import tensorflow as tf
import numpy as np
from fastkde import fastKDE
from numpy import *
from sklearn.preprocessing import scale
import csv





def getHisto(x, y, size, maxstd):

    intx = np.round(x * (size) / maxstd / 2)
    inty = np.round(y * (size) / maxstd / 2)

    pXY = np.zeros((size, size))

    for i in range(len(x)):
        coordx = int(min(intx[i], size / 2 - 1) + size / 2)
        coordy = int(min(inty[i], size / 2 - 1) + size / 2)

        pXY[coordx, coordy] += 1

    maxPXY = amax(pXY)

    pXY = pXY / maxPXY

    return pXY



def featurizePairs(path, filepairs, filetargets, maxstd, size, featurizingmethod = 0, ratio=1, doubletrainset=True):

    f = open(path + filepairs );
    pairs = f.readlines();
    pairs.pop(0)
    f.close();

    y_te = np.genfromtxt(path + filetargets, delimiter=",")

    for k in range(0, int(len(y_te)*ratio)):

        if(k%100==0):
            print(k)

        r = pairs[k].split(",", 2)

        x = scale(np.array(r[1].split(), dtype=np.float))
        y = scale(np.array(r[2].split(), dtype=np.float))

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

            arrayXY = np.ravel(pXY)

            if(k==0):
                vectorizedPairs = arrayXY
            else:
                vectorizedPairs = np.vstack((vectorizedPairs, arrayXY))

            if(doubletrainset == True):
                arrayYX = np.ravel(np.transpose(pXY))
                vectorizedPairs = np.vstack((vectorizedPairs, arrayYX))

            if y_te[k] == -1:
                arrayTargetXY = np.array([1,0,0])
                arrayTargetYX = np.array([0,0,1])

            elif y_te[k] == 0:
                arrayTargetXY = np.array([0,1,0])
                arrayTargetYX = np.array([0,1,0])

            elif y_te[k] == 1:
                arrayTargetXY = np.array([0,0,1])
                arrayTargetYX = np.array([1,0,0])

            if(k==0):
                vectorizedTarget = arrayTargetXY
            else:
                vectorizedTarget = np.vstack((vectorizedTarget, arrayTargetXY))

            if (doubletrainset == True):
                vectorizedTarget = np.vstack((vectorizedTarget, arrayTargetYX))

        except ValueError:
            print("pbkde nbpoints pairs " + str(k))


    np.savetxt(path + "vectorized" + "_maxstd" + maxstd + "_size" + size + filepairs, vectorizedPairs)
    np.savetxt(path + "vectorized" + "_maxstd" + maxstd + "_size" + size + filetargets, vectorizedTarget)

    return vectorizedPairs,vectorizedTarget




def getNextBatch(batch_size):

    global _index_in_epoch
    global _num_examples
    global _epochs_completed
    global trainpairs
    global traintargets

    start = _index_in_epoch
    _index_in_epoch += batch_size

    newEpoch = False
    if _index_in_epoch > _num_examples:
          # Finished epoch
      _epochs_completed += 1
      print("_epochs_completed " + str(_epochs_completed))
      newEpoch = True
      # Shuffle the data
      perm = np.arange(_num_examples)
      np.random.shuffle(perm)
      trainpairs = trainpairs[perm]
      traintargets = traintargets[perm]
      # Start next epoch
      start = 0
      _index_in_epoch = batch_size
      assert batch_size <= _num_examples

    end = _index_in_epoch
    return trainpairs[start:end], traintargets[start:end], newEpoch


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1], padding='SAME')



# featurize parameters
alreadyFeaturized = True
size = 64               # Image size x size
maxstd = 2              # max std for outliers removal
ratio = 0.1               # ratio training set we keep
doubletrainset = True   # duplicate training set
featurizingmethod = 0   # 0 for histo, 1 for kernel

# network parameters
sizepatchX = 8
sizepatchY = 8
sizehiddenlayer = 256 # sizehiddenlayer = 1024

# training parameters
sizeBatch = 50
nbBatch = 150000

keep_probDropout = 0.5
regularizationparam = 0.01

path = "data/"

# source file paths
filetrainpairs = "train_pairs.csv"
filetraintargets = "train_target.csv"
filevalidpairs = "CEfinal_valid_pairs.csv"
filevalidtargets = "CEfinal_valid_target.csv"

# image file paths
filetrainpairsvectorized = "data/vectorizedtrain_numpoints64_maxstd2.csv"
filetraintargetsvectorized = "data/vectorizedtraintarget.csv"
filevalidpairsvectorized = "data/vectorizedvalid_numpoints64_maxstd2.csv"
filevalidtargetsvectorized = "data/vectorizedvalidtarget.csv"

# filetrainpairsvectorized = path + "vectorizedtrain_pairs.csv"
# filetraintargetsvectorized = path + "vectorizedtrain_target.csv"
# filevalidpairsvectorized = path + "vectorizedCEfinal_valid_pairs.csv"
# filevalidtargetsvectorized = path + "vectorizedCEfinal_valid_target.csv"


if(alreadyFeaturized == True):
    print("Load data")
    validpairs = np.loadtxt(filevalidpairsvectorized)
    validtarget = np.loadtxt(filevalidtargetsvectorized)
    print("Valid data loaded")
    trainpairs = np.loadtxt(filetrainpairsvectorized)
    traintargets = np.loadtxt(filetraintargetsvectorized)
    print("Train data loaded")
else :
    print("Featurize train data")
    trainpairs,traintargets = featurizePairs(path, filetrainpairs, filetraintargets, maxstd, size, featurizingmethod,ratio, doubletrainset)
    print("Featurize validation data")
    validpairs,validtarget = featurizePairs(path, filevalidpairs, filevalidtargets, maxstd, size, featurizingmethod, ratio, doubletrainset)




print("Start learning")
_num_examples = trainpairs.shape[0]
print("nb examples for training : " + str(_num_examples))
_index_in_epoch = 0
_epochs_completed = 0

sess = tf.InteractiveSession()

# Convolutionl network creation
W_conv1 = weight_variable([sizepatchX, sizepatchY, 1, 32])
b_conv1 = bias_variable([32])

x = tf.placeholder(tf.float32, shape=[None, size**2])
y_ = tf.placeholder(tf.float32, shape=[None, 3])

x_image = tf.reshape(x, [-1,size,size,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([sizepatchX, sizepatchY, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

newsize = int(size/4)

W_fc1 = weight_variable([newsize * newsize * 64, sizehiddenlayer])
b_fc1 = bias_variable([sizehiddenlayer])

h_pool2_flat = tf.reshape(h_pool2, [-1, newsize*newsize*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([sizehiddenlayer, 3])
b_fc2 = bias_variable([3])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_)
#                                + regularizationparam*(tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1)+ tf.nn.l2_loss(W_fc2)))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

for i in range(nbBatch):

  batch = getNextBatch(sizeBatch)

  # End epoch
  # if(batch[2] ==True):
  #     print("test accuracy validation %g" % accuracy.eval(feed_dict={x: validpairs, y_: validtarget, keep_prob: 1.0}))

  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))

  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: keep_probDropout})


print("test accuracy %g"%accuracy.eval(feed_dict={x: validpairs, y_: validtarget, keep_prob: 1.0}))




