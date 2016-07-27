'''
Applying Clustering for multiple number of clusters and calculating the Misclassification error
Author : Diviyan Kalainathan
Date : 25/06/2016

'''

import kmeans_w_plot as kmeans
import performance_evaluation
import os, numpy,sys

# n_clusters=[10,15,20,30,40,50,70,90,110,140,170,200]
n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
n_init = 3000
batch_size = 8000
n_runs = 20
n_iter = 300
paths = ['ao','ao_vp','as','as_vp','wo','wo_vp','ws','ws_vp']

input_data = ['input/ao+/computed_data_a_o+5.csv',
              'input/ao+/computed_data_a_ovp+5.csv',
              'input/as+/computed_data_a_s+3.csv',
              'input/as+/computed_data_a_svp+3.csv',
              'input/wo+/computed_data_w_o+_5.csv',
              'input/wo+/computed_data_w_ovp+_5.csv',
              'input/ws+/computed_data_w_s+_5.csv',
              'input/ws+/computed_data_w_svp+_5.csv',
              ]

dimensions = [5,5,3,3,5,5,3,3]

itr= int(sys.argv[1])

data_files = input_data[itr]
for i in n_clusters:
    print 'num of clusters : ', i
    path = paths[itr] + str(i)
    if not os.path.exists('output/' + path):
        os.makedirs('output/' + path)

    kmeans.minibatch_kmeans(i, n_init, batch_size, path, data_files, dimensions[itr], n_runs, 'random', n_iter)
    dist = []
    for k in range(0, n_runs - 1):
        for j in range(k + 1, n_runs):
            dist += [performance_evaluation.distance_clustering(path, k, j, i, n_init)]

    print dist
    numpy.savetxt('output/' + path + '/resultdist_' + str(i) + '.csv', dist, delimiter=';')


# Kmeans++

n_init = 500

data_files = input_data[itr]
for i in n_clusters:
    print 'num of clusters : ', i
    path = paths[itr] + 'km++_' + str(i)
    if not os.path.exists('output/' + path):
        os.makedirs('output/' + path)

    kmeans.minibatch_kmeans(i, n_init, batch_size, path, data_files, dimensions[itr], n_runs, 'k-means++',
                            n_iter)
    dist = []
    for k in range(0, n_runs - 1):
        for j in range(k + 1, n_runs):
            dist += [performance_evaluation.distance_clustering(path, k, j, i, n_init)]

    print dist
    numpy.savetxt('output/' + path + '/resultdist_' + str(i) + '.csv', dist, delimiter=';')

