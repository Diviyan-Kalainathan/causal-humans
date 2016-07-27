'''
Kmeans algorithm on dimension reduced data
Author : Diviyan Kalainathan
Date : 6/06/2016
#DEPRECATED
'''

import numpy
from sklearn.cluster import KMeans,DBSCAN
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from random import sample
from collections import Counter
import matplotlib.colors as colors
from random import randint
import colorsys






numpy.set_printoptions(threshold='nan')


inputdata = numpy.loadtxt('input/computed_data9dim-md.csv',delimiter=';')

inputdata=numpy.transpose(inputdata)
print(numpy.shape(inputdata))
result = DBSCAN()
#n_clusters=n_clusters,n_init=300,n_jobs=-2,init='random'
R= result.fit_predict(inputdata)
R2 = result.components_
print(R)
print(numpy.shape(R2))

'''
whitened=whiten(inputdata)

cdbook,distortion= kmeans(whitened,300,iter=30)

print(cdbook)
print(distortion)'''
n_clusters=30


N = n_clusters
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)


randomdata=numpy.zeros((5000,9))
rand_csei=numpy.zeros((5000,1))
randlist= sample(range(32693),5000)


c_csei = numpy.loadtxt('input/counter_csei.csv', delimiter=';')

for i in range(0,5000):
    randomdata[i,:]=inputdata[randlist[i],:]
    rand_csei[i]= c_csei[randlist[i]]
displaydata = numpy.vstack((randomdata,R2))

print(numpy.shape(displaydata))


model = TSNE(n_components=2)
numpy.set_printoptions(suppress=True)
toprint= model.fit_transform(displaydata)



for i in range (0,len(toprint[:,1])):
    if i <5000:
        '''
        if rand_csei[i]==1:
            color='black'
        elif rand_csei[i]==2:
            color='blue'
        elif rand_csei[i] == 3:
            color = 'beige'
        elif rand_csei[i] == 4:
            color = 'salmon'
        elif rand_csei[i] == 5:
            color = 'cyan'
        elif rand_csei[i] == 6:
            color = 'crimson'
        elif rand_csei[i] == 8:
            color = 'grey'
        elif rand_csei[i] == 9:
            color = 'green'
        elif rand_csei[i] == 10:
            color = 'ivory'
        elif rand_csei[i] == 11:
            color = 'limegreen'
        elif rand_csei[i] == 12:
            color = 'maroon'
        elif rand_csei[i] == 13:
            color = 'orange'
        elif rand_csei[i] == 14:
            color = 'purple'
        elif rand_csei[i] == 15:
            color = 'red'
        elif rand_csei[i] == 16:
            color = 'salmon'
        elif rand_csei[i] == 17:
            color = 'yellow'
        elif rand_csei[i] == 18:
            color = 'ivory'
        else:
            color='snow'''


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
plt.bar(range(n_clusters),s_values)
plt.show()


numpy.savetxt('output/cluster_predictions_80s-md.csv', R, delimiter=';')
