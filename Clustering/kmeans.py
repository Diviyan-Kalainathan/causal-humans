'''
Kmeans algorithm on dimension reduced data - Algorithm : mini-batch k-means
Author : Diviyan Kalainathan
Date : 6/06/2016
#DEPRECATED
'''

import numpy
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from random import sample
from collections import Counter
import colorsys


def minibatch_kmeans(num_clusters, num_init, size_batch, folder_name,
                     data_file, num_runs=1, type_init='random', num_iter=100):
    """
    :param num_clusters: Number of clusters to be found but the algorithm (int)
    :param num_init: number of different runs of the algorithm (int)
    :param size_batch: Number of data values per batch(int)
    :param folder_name: Folder in which the results are stored (String)
    :param data_file: Name of the input file for the minibatch k-means
    :param num_runs: Number of runs of the whole program (int)
    :param type_init: Type of init for the Kmeans algorithm (String)
    :param num_iter: Number of max iterations for the Kmeans algorithm(int)
    :return:
    """

    for run in range(0, num_runs):

        N = num_clusters

        # Creating colors to plot according to which cluster the points belong to
        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

        numpy.set_printoptions(threshold='nan')

        # Preparing the data for k-means
        inputdata = numpy.loadtxt(data_file, delimiter=';')

        inputdata = numpy.transpose(inputdata)
        inputdata = numpy.hstack((inputdata, numpy.arange(numpy.shape(inputdata)[0])[:, numpy.newaxis]))
        inputdata = numpy.random.permutation(inputdata)
        index_array = inputdata[:, -1:]
        inputdata = inputdata[:, :-1]
        print(numpy.shape(inputdata))

        # Applying the algorithm
        result = MiniBatchKMeans(n_clusters=num_clusters, n_init=num_init, init=type_init, batch_size=size_batch,
                                 verbose=True, max_iter=num_iter)
        R = result.fit_predict(inputdata)
        R2 = result.cluster_centers_
        print(R)
        print(numpy.shape(R2))

        # For the first run, creating plots
        if run == 0:

            # TSNE
            # For t-SNE, choosing 4000 random points to plot
            randomdata = numpy.zeros((4000, 16))  # 16 is the dimensionality of the data
            rand_csei = numpy.zeros((4000, 1))
            randlist = sample(range(31112), 4000)

            c_csei = numpy.loadtxt('input/counter_csei.csv', delimiter=';')

            for i in range(0, 4000):
                randomdata[i, :] = inputdata[randlist[i], :]
                rand_csei[i] = c_csei[randlist[i]]
            displaydata = numpy.vstack((randomdata, R2))

            print(numpy.shape(displaydata))

            model = TSNE(n_components=2)
            numpy.set_printoptions(suppress=True)
            toprint = model.fit_transform(displaydata)

            for i in range(0, len(toprint[:, 1])):
                if i < 4000:

                    color = RGB_tuples[R[randlist[i]]]

                    plt.plot(toprint[i, 0], toprint[i, 1], 'x', color=color)




                else:
                    plt.plot(toprint[i, 0], toprint[i, 1], 'rD')

            plt.savefig(
                'output/' + str(folder_name) + '/t-sne_c' + str(num_clusters) + '_n' + str(num_init) + '_r' + str(
                    run) + '.pdf')
            plt.clf()

        # Plotting histogram of repartition of points in clusters
        print(type(R))
        R_s = R.flatten()
        R_s.sort()
        labels, values = zip(*Counter(R_s).items())
        s_values = sorted(values)
        print(s_values)
        try:
            plt.bar(range(num_clusters), s_values)
            plt.savefig(
                'output/' + str(folder_name) + '/hist_c' + str(num_clusters) + '_n' + str(num_init) + '_r' + str(
                    run) + '.pdf')
        except:
            print('Plot failed')

        plt.clf()

        # Save data
        print(numpy.shape(R))
        print(numpy.shape(index_array))
        R = R[:, numpy.newaxis]
        R = numpy.hstack((R, index_array))
        print(numpy.shape(R))
        numpy.savetxt('output/' + folder_name + '/cluster_centers_c' + str(num_clusters)
                      + '_n' + str(num_init) + '_r' + str(run) + '.csv', R2, delimiter=';')
        numpy.savetxt('output/' + folder_name + '/cluster_predictions_c' + str(num_clusters)
                      + '_n' + str(num_init) + '_r' + str(run) + '.csv', R, delimiter=';')

    return 0
