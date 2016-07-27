'''
Generates plot using Clustered data
Author : Diviyan Kalainathan
Date : 24/06/2016
'''
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import numpy, csv

mode = 2  # 1 for errorbars fo misclassification
# 2 for TSNE

if mode == 1:
    # n_clusters=[10,15,20,30,40,50,70,90,110,140,170,200]
    n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
    data = numpy.zeros((len(n_clusters), 190))
    mean_values = numpy.zeros((len(n_clusters)))
    std_values = numpy.zeros((len(n_clusters)))
    # Load data
    j = 0
    for i in n_clusters:
        data[j, :] = numpy.loadtxt('output/wo/dist/rand/resultdist_' + str(i) + '.csv', delimiter=';')
        mean_values[j] = numpy.mean(data[j, :])
        std_values[j] = numpy.std(data[j, :])
        j += 1

    plt1 = plt.errorbar(n_clusters, mean_values, std_values)
    j = 0
    for i in n_clusters:
        data[j, :] = numpy.loadtxt('output/wo/dist/rand_vp/resultdist_' + str(i) + '.csv', delimiter=';')
        mean_values[j] = numpy.mean(data[j, :])
        std_values[j] = numpy.std(data[j, :])
        j += 1

    plt2 = plt.errorbar(n_clusters, mean_values, std_values)

    j = 0
    for i in n_clusters:
        data[j, :] = numpy.loadtxt('output/wo/dist/k++/resultdist_' + str(i) + '.csv', delimiter=';')
        mean_values[j] = numpy.mean(data[j, :])
        std_values[j] = numpy.std(data[j, :])
        j += 1

    plt3 = plt.errorbar(n_clusters, mean_values, std_values)
    j = 0

    for i in n_clusters:
        data[j, :] = numpy.loadtxt('output/wo/dist/k++_vp/resultdist_' + str(i) + '.csv', delimiter=';')
        mean_values[j] = numpy.mean(data[j, :])
        std_values[j] = numpy.std(data[j, :])
        j += 1

    plt4 = plt.errorbar(n_clusters, mean_values, std_values)
    # Legend

    plt.title('Distance between clustering - Objective - Filtered questions, 20 runs at each step')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distance')
    plt.legend([plt1, plt2, plt3, plt4], ['Random K-means',
                                          'Random K-means with sqrt(eig-values)',
                                          'K-means++',
                                          'K-means++, with sqrt(eig-values)'])

elif mode == 2:

    with open('output/ws/km++/wskm++_4/tsne_c4_n500_r0.csv', 'rb') as datafile:
        var_reader = csv.reader(datafile, delimiter=';', quotechar='|')
        header = next(var_reader)
        for row in var_reader:
            if var_reader.line_num < 4002:
                plt.plot(row[0], row[1], 'x', color=eval(row[2]))
            else:
                plt.plot(row[0], row[1], 'rD')

plt.show()
