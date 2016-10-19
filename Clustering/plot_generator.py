'''
Generates plot using Clustered data
Author : Diviyan Kalainathan
Date : 24/06/2016
'''
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import numpy, csv

mode = 2 # 1 for graph of misclassification
# 2 for TSNE

if mode == 1:
    # n_clusters=[10,15,20,30,40,50,70,90,110,140,170,200]
    n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]#, 18, 20]
    data = numpy.zeros((len(n_clusters), 4950))
    mean_values = numpy.zeros((len(n_clusters)))
    std_values = numpy.zeros((len(n_clusters)))
    # Load data
    j = 0
    for i in n_clusters:
        #with open('output/idf/dist/as/resultdist_ari' + str(i) + '.csv', delimiter=';') as inputfile:

        data[j, :] = numpy.loadtxt('output/idf/dist/as/resultdist_ari' + str(i) + '.csv', delimiter=';')
        mean_values[j] = numpy.mean(data[j, :])
        std_values[j] = numpy.std(data[j, :])
        j += 1

    print(data[0,:])
    print(mean_values)

    plt1 = plt.errorbar(n_clusters, mean_values, std_values)
    j = 0
    for i in n_clusters:
        data[j, :] = numpy.loadtxt('output/idf/dist/as_vp/resultdist_ari' + str(i) + '.csv', delimiter=';')
        mean_values[j] = numpy.mean(data[j, :])
        std_values[j] = numpy.std(data[j, :])
        j += 1

    plt2 = plt.errorbar(n_clusters, mean_values, std_values)

    j = 0
    for i in n_clusters:
        data[j, :] = numpy.loadtxt('output/idf/dist/ws/resultdist_ari' + str(i) + '.csv', delimiter=';')
        mean_values[j] = numpy.mean(data[j, :])
        std_values[j] = numpy.std(data[j, :])
        j += 1

    plt3 = plt.errorbar(n_clusters, mean_values, std_values)
    j = 0

    for i in n_clusters:
        data[j, :] = numpy.loadtxt('output/idf/dist/ws_vp/resultdist_ari' + str(i) + '.csv', delimiter=';')
        mean_values[j] = numpy.mean(data[j, :])
        std_values[j] = numpy.std(data[j, :])
        j += 1

    plt4 = plt.errorbar(n_clusters, mean_values, std_values)
    # Legend

    plt.title('Performance du clustering - Questions subjectives -  100 iterations par point')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Performance')
    plt.legend([plt1, plt2, plt3, plt4], ['1 - a','2 - a_vp','3 - w','4 - w_vp'],loc=1)
    '''['Random K-means',
      'Random K-means with sqrt(eig-values)',
      'K-means++',
      'K-means++, with sqrt(eig-values)'])'''


elif mode == 2:

    with open('output/idf/ws/wskm++_6/tsne_c6_n500_r0.csv', 'rb') as datafile:
        var_reader = csv.reader(datafile, delimiter=';', quotechar='|')
        header = next(var_reader)
        for row in var_reader:
            if var_reader.line_num < 4002:
                plt.plot(row[0], row[1], '.', color=eval(row[2]))
            else:
                plt.plot(row[0], row[1], 'rD')

plt.show()
