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
n_init = 1# 3000
batch_size = 8000
n_runs = 100#20
n_iter = 300
paths = ['ao','ao_vp','as','as_vp','wo','wo_vp','ws','ws_vp','wa6','wa6_vp','wa12','wa12_vp','aa','aa_vp']

input_data = ['input/ao~/computed_data_ao_~9.csv',
              'input/ao~/computed_data_ao_vp~9.csv',
              'input/as~/computed_data_as_~4.csv',
              'input/as~/computed_data_as_vp~4.csv',
              'input/wo~/computed_data_wo_~8.csv',
              'input/wo~/computed_data_wo_vp~8.csv',
              'input/ws~/computed_data_ws_~5.csv',
              'input/ws~/computed_data_ws_~5.csv',
              'input/wa~/computed_data_wa_~6.csv',
              'input/wa~/computed_data_wa_vp~6.csv',
              'input/wa~/computed_data_wa_~12.csv',
              'input/wa~/computed_data_wa_vp~12.csv',
              'input/aa~/computed_data_aa_~10.csv',
              'input/aa~/computed_data_aa_~vp10.csv'
              ]

dimensions = [9,9,4,4,8,8,5,5,6,6,12,12,10,10]

itr= int(sys.argv[1])

data_files = input_data[itr]
for i in n_clusters:
    print 'Num of clusters : ', i
    path = paths[itr]+'/'+paths[itr] + str(i)

    if not os.path.exists('output/dist/' + paths[itr] + '/'):
        os.makedirs('output/dist/' + paths[itr] + '/')
    if not os.path.exists('output/' + path):
        os.makedirs('output/' + path)

    kmeans.minibatch_kmeans(i, n_init, batch_size, path, data_files, dimensions[itr], n_runs, 'random', n_iter)
    dist = []
    for k in range(0, n_runs - 1):
        for j in range(k + 1, n_runs):
            dist += [performance_evaluation.Clustering_performance_evaluation(2,path, k, j, i, n_init)]

    numpy.savetxt('output/dist/' + paths[itr] + '/resultdist_ari' + str(i) + '.csv', dist, delimiter=';')

    dist = []
    for k in range(0, n_runs - 1):
        for j in range(k + 1, n_runs):
            dist += [performance_evaluation.Clustering_performance_evaluation(3, path, k, j, i, n_init)]

    numpy.savetxt('output/dist/' + paths[itr] + '/resultdist_vm' + str(i) + '.csv', dist, delimiter=';')


# Kmeans++

n_init = 1 #500

data_files = input_data[itr]
for i in n_clusters:
    print 'Num of clusters : ', i
    path = paths[itr]+'/'+paths[itr]+'_km++' + str(i)
    if not os.path.exists('output/' + path):
        os.makedirs('output/' + path)

    kmeans.minibatch_kmeans(i, n_init, batch_size, path, data_files, dimensions[itr], n_runs, 'k-means++',
                            n_iter)
    dist = []
    for k in range(0, n_runs - 1):
        for j in range(k + 1, n_runs):
            dist += [performance_evaluation.Clustering_performance_evaluation(2,path, k, j, i, n_init)]

    numpy.savetxt('output/dist/' + paths[itr] + '/resultdistkm_ari' + str(i) + '.csv', dist, delimiter=';')

    dist = []

    for k in range(0, n_runs - 1):
        for j in range(k + 1, n_runs):
            dist += [performance_evaluation.Clustering_performance_evaluation(3, path, k, j, i, n_init)]

    numpy.savetxt('output/dist/' + paths[itr] + '/resultdistkm_vm' + str(i) + '.csv', dist, delimiter=';')

