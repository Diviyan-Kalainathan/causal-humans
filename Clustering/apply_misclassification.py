'''
Calculating the Misclassification error
Author : Diviyan Kalainathan
Date : 23/07/2016

'''

import performance_evaluation
import os, numpy,sys
from sklearn import metrics

# n_clusters=[10,15,20,30,40,50,70,90,110,140,170,200]
n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
n_init = 3000
batch_size = 8000
n_runs = 20
n_iter = 300
paths = ['ao','ao_vp','as','as_vp','wo','wo_vp','ws','ws_vp']

mode = 1
#1 : Dist by Marina Meila
#2 : Dist

for itr in range(len(paths)):
    print 'itr : ', itr
    for i in n_clusters:
        print ' - number of clusters : ', i

        n_init = 3000

        path = paths[itr] + str(i)
        if not os.path.exists('output/' + path):
            os.makedirs('output/' + path)

        dist = []
        for k in range(0, n_runs - 1):
            for j in range(k + 1, n_runs):
                dist += [performance_evaluation.Clustering_performance_evaluation(mode,path, k, j, i, n_init)]

        print dist
        numpy.savetxt('output/' + path + '/resultdist_' + str(i) + '.csv', dist, delimiter=';')

        n_init=500
        path = paths[itr]+'km++_' + str(i)


        dist = []
        for k in range(0, n_runs - 1):
            for j in range(k + 1, n_runs):
                dist += [performance_evaluation.Clustering_performance_evaluation(mode,path, k, j, i, n_init)]

        print dist
        numpy.savetxt('output/' + path + '/resultdist_' + str(i) + '.csv', dist, delimiter=';')


