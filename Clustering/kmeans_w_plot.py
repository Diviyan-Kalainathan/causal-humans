'''
Kmeans algorithm on dimension reduced data without plots
Author : Diviyan Kalainathan
Date : 6/06/2016

'''

import numpy
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from random import sample
from collections import Counter
import colorsys, csv


def minibatch_kmeans(num_clusters, num_init, size_batch, folder_name,
                     data_file, datadim, num_runs=1, type_init='random', num_iter=100):
    """
    :param num_clusters: Number of clusters to be found but the algorithm (int)
    :param num_init: number of different runs of the algorithm (int)
    :param size_batch: Number of data values per batch(int)
    :param folder_name: Folder in which the results are stored
    :param data_file: Name of the input file for the minibatch k-means(String)
    :param datadim: dimensionality of the input data(int)
    :param num_runs: Number of runs of the whole program (int)
    :param type_init: Type of init for the Kmeans algorithm (String)
    :param num_iter: Number of max iterations for the Kmeans algorithm(int)
    :return:
    """

    for run in range(0, num_runs):

        N = num_clusters
        print N,'--run n : ', run

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

        # Applying the algorithm

        result = MiniBatchKMeans(n_clusters=num_clusters, n_init=num_init, init=type_init, batch_size=size_batch,
                                 max_iter=num_iter)
        R = result.fit_predict(inputdata)
        R2 = result.cluster_centers_

        # For the first run, creating plots

        if False:#run == 0:

            # TSNE
            # For t-SNE, choosing 4000 random points to plot
            randomdata = numpy.zeros((4000, datadim))
            randlist = sample(range(31112), 4000)

            for i in range(0, 4000):
                randomdata[i, :] = inputdata[randlist[i], :]
            displaydata = numpy.vstack((randomdata, R2))

            model = TSNE(n_components=2)
            numpy.set_printoptions(suppress=True)
            toprint = model.fit_transform(displaydata)

            with open('output/' + folder_name + '/tsne_c' + str(num_clusters) +
                              '_n' + str(num_init) + '_r' + str(run) + '.csv', 'wb') as outputfile:
                datawriter = csv.writer(outputfile, delimiter=';', quotechar='|')
                datawriter.writerow(
                    str(folder_name) + '_t-sne_c' + str(num_clusters) + '_n' + str(num_init) + '_r' + str(run))
            for i in range(0, len(toprint[:, 1])):
                point_toprint = []
                if i < 4000:

                    color = RGB_tuples[R[randlist[i]]]
                    point_toprint += [toprint[i, 0]]
                    point_toprint += [toprint[i, 1]]
                    point_toprint += [color]

                else:
                    point_toprint += [toprint[i, 0]]
                    point_toprint += [toprint[i, 1]]
                    #

                with open('output/' + folder_name + '/tsne_c' + str(num_clusters) +
                                  '_n' + str(num_init) + '_r' + str(run) + '.csv', 'a') as outputfile:
                    datawriter = csv.writer(outputfile, delimiter=';', quotechar='|',
                                            lineterminator='\n')
                    datawriter.writerow(point_toprint)

            # Plotting histogram of repartition of points in clusters

            R_s = R.flatten()
            R_s.sort()
            labels, values = zip(*Counter(R_s).items())
            s_values = sorted(values)
            numpy.savetxt('output/' + folder_name + '/hist_c' + str(num_clusters) +
                          '_n' + str(num_init) + '_r' + str(run) + '.csv', s_values, delimiter=';')

        # Save data


        R = R[:, numpy.newaxis]
        R = numpy.hstack((R, index_array))
        numpy.savetxt('output/' + folder_name + '/cluster_centers_c' + str(num_clusters)
                      + '_n' + str(num_init) + '_r' + str(run) + '.csv', R2, delimiter=';')
        numpy.savetxt('output/' + folder_name + '/cluster_predictions_c' + str(num_clusters)
                      + '_n' + str(num_init) + '_r' + str(run) + '.csv', R, delimiter=';')

    return 0
