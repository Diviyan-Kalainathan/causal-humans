'''
Kmeans algorithm on dimension reduced data
Author : Diviyan Kalainathan
Date : 6/06/2016

'''

import numpy
from sklearn.cluster import KMeans,AgglomerativeClustering
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from random import sample
from collections import Counter
import matplotlib.colors as colors
from random import randint
import colorsys

num_clusters=120

numpy.set_printoptions(threshold='nan')


inputdata = numpy.loadtxt('input/computed_data9dim-md.csv',delimiter=';')

inputdata=numpy.transpose(inputdata)

print(numpy.shape(inputdata))

#R2 = result.components_
#print(R)

'''
whitened=whiten(inputdata)

cdbook,distortion= kmeans(whitened,300,iter=30)

print(cdbook)
print(distortion)'''


N = num_clusters
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)


randomdata=numpy.zeros((5000,9))
rand_csei=numpy.zeros((5000,1))
randlist= sample(range(5000),5000)


c_csei = numpy.loadtxt('input/counter_csei.csv', delimiter=';')

for i in range(0,5000):
    randomdata[i,:]=inputdata[randlist[i],:]
    rand_csei[i]= c_csei[randlist[i]]
displaydata = randomdata #numpy.vstack((randomdata,R2))

print(numpy.shape(displaydata))
result = AgglomerativeClustering(n_clusters=num_clusters)
#num_clusters=num_clusters,n_init=300,n_jobs=-2,init='random'
R= result.fit_predict(displaydata)
print(R)

model = TSNE(n_components=2)
numpy.set_printoptions(suppress=True)
toprint= model.fit_transform(displaydata)



for i in range (0,len(toprint[:,1])):
    if i <5000:

        color=RGB_tuples[R[randlist[i]]]



        plt.plot(toprint[i, 0], toprint[i, 1], 'x' ,color=color)




    else:
        plt.plot(toprint[i, 0], toprint[i, 1], 'rD')

plt.show()
print(type(R))
R_s=R.flatten()
R_s.sort()
labels, values = zip(*Counter(R_s).items())
s_values=sorted(values)
print(s_values)
plt.bar(range(num_clusters),s_values)
plt.show()


print(numpy.shape(R))
R=R[:, numpy.newaxis]
R=numpy.hstack((R,range(len(R))))
print(numpy.shape(R))
numpy.savetxt('output/cluster_predictions_50-md++2.csv', R, delimiter=';')
