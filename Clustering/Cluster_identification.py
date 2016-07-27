'''
Identification of clusters of 2 different clustering runs
Author : Diviyan Kalainathan
Date : 6/06/2016
#DEPRECATED
'''

import numpy

numpy.set_printoptions(threshold='nan')

data_1 = numpy.loadtxt('output/orig50/cluster_predictions_c50_n5000_r5.csv', delimiter=';')
data_2 = numpy.loadtxt('output/orig50/cluster_predictions_c50_n5000_r8.csv', delimiter=';')

data_1 = numpy.asarray(sorted(data_1, key=lambda x: x[0]))
data_2 = numpy.asarray(sorted(data_2, key=lambda x: x[1]))
print(numpy.shape(data_1))

n_clusters = len(set(data_1[:, 0]))
data_set_length = len(data_1[:, 0])

print(data_set_length)
print(n_clusters)
print(data_2[1, 1])

cluster_ident = numpy.zeros((n_clusters))
cluster_ident_2 = numpy.zeros((n_clusters))

cluster_accuracy = numpy.zeros((n_clusters))
cluster_accuracy_2 = numpy.zeros((n_clusters))

accuracy_matrix_1 = numpy.zeros((n_clusters, n_clusters))
accuracy_matrix_2 = numpy.zeros((n_clusters, n_clusters))  # The other way around : accuracy of 2 according to 1

# row_number
j = 0

for i in range(0, n_clusters):
    cluster_count = numpy.zeros((n_clusters))

    while j < data_set_length and data_1[j, 0] == i:
        cluster_count[data_2[data_1[j, 1], 0]] += 1
        j += 1

    accuracy_matrix_1[i, :] = cluster_count / numpy.sum(cluster_count)
    cluster_ident[i] = numpy.argmax(cluster_count)
    cluster_accuracy[i] = cluster_count[cluster_ident[i]] / numpy.sum(cluster_count)

accuracy_1 = numpy.mean(cluster_accuracy)

data_1 = numpy.asarray(sorted(data_1, key=lambda x: x[1]))
data_2 = numpy.asarray(sorted(data_2, key=lambda x: x[0]))
print(data_1[0:10, 1])
print(data_2[0:10, 0])

j = 0

for i in range(0, n_clusters):
    cluster_count = numpy.zeros((n_clusters))

    while j < data_set_length and data_2[j, 0] == i:
        cluster_count[data_1[data_2[j, 1], 0]] += 1
        j += 1

    accuracy_matrix_2[i, :] = cluster_count / numpy.sum(cluster_count)
    cluster_ident_2[i] = numpy.argmax(cluster_count)
    cluster_accuracy_2[i] = cluster_count[cluster_ident_2[i]] / numpy.sum(cluster_count)

accuracy_2 = numpy.mean(cluster_accuracy_2)

total_accuracy = numpy.mean((accuracy_1, accuracy_2))

# print(accuracy_matrix_1)
print('-' * 70)
# print(accuracy_matrix_2)
print('-' * 70)
print('Cluster index 1: ')
print(cluster_ident)
print('-' * 70)
print('Cluster accuracy 1: ')
print(cluster_accuracy)
print('-' * 70)
print('Cluster index 2: ')
print(cluster_ident_2)
print('-' * 70)
print('Cluster accuracy 2: ')
print(cluster_accuracy_2)
print('-' * 70)
print('Total accuracy : ' + repr(total_accuracy))
