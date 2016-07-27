'''
Computing the misclassification error distance between to 2 k-means clustering
according to Marina Meila, "The Uniqueness of a Good Optimum for K-Means", ICML 2006
Author : Diviyan Kalainathan
Date : 20/06/2016

'''

import csv,numpy,itertools
from sklearn import metrics

def Clustering_performance_evaluation(mode, folder_name, run1, run2, num_clusters, num_init):
    """
    :param mode: selects which distance is to be used
    :param folder_name: Folder of the runs (String)
    :param run1: Number of the run 1 (int)
    :param run2: Number of the run 2 (int)
    :param num_clusters:
    :param num_init:
    :return:  distance value (float?)

    """
    numpy.set_printoptions(threshold='nan')

    valid_data= True

    #Checking if the data is valid by loading & testing the shape of it
    try:
        data_1=numpy.loadtxt('output/'+folder_name+'/cluster_predictions_c'+ str(num_clusters)
                      + '_n'+ str(num_init) +'_r'+ str(run1)+'.csv',delimiter=';')
        data_2=numpy.loadtxt('output/'+folder_name+'/cluster_predictions_c'+ str(num_clusters)
                      + '_n'+ str(num_init) +'_r'+ str(run2)+'.csv',delimiter=';')

        if data_1.shape != data_2.shape:
            valid_data=False

    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
        valid_data=False


    if valid_data:
        n_samples=data_1.shape[0]
        data_1 = numpy.asarray(sorted(data_1, key=lambda x: x[1]))
        data_2 = numpy.asarray(sorted(data_2, key=lambda x: x[1]))

        if mode==1:
            #Distance defined by Marina Meila : k! complexity
            clustering_1=numpy.zeros((n_samples,num_clusters))
            clustering_2=numpy.zeros((n_samples,num_clusters))

            for x in range(0,n_samples):
                clustering_1[x,data_1[x,0]]+=1
                clustering_2[x,data_2[x,0]]+=1

            '''for y in range(0,num_clusters):
                try:
                    clustering_1[:,y]*=1/numpy.sqrt(numpy.sum(clustering_1[:,y]))
                except ZeroDivisionError:
                    clustering_1[:,y]=0
                try:
                    clustering_2[:,y]*=1/numpy.sqrt(numpy.sum(clustering_2[:,y]))
                except ZeroDivisionError:
                    clustering_2[:,y]=0
            ''' # No normalisation needed


            confusion_matrix=numpy.dot(numpy.transpose(clustering_1),clustering_2)
            max_confusion=0
            result = []

            for perm in itertools.permutations(range(num_clusters)):
                confusion=0
                for i in range(0, num_clusters):
                    confusion += confusion_matrix[i, perm[i]]

                if max_confusion<confusion:
                    max_confusion=confusion


            distance=(max_confusion/n_samples)
            return distance


        elif mode==2:
            #Ajusted rand index
            distance=metrics.adjusted_rand_score(data_1[:,0],data_2[:,0])

            return distance

        elif mode==3:
            #V-mesure
            distance=metrics.v_measure_score(data_1[:,0],data_2[:,0])

            return distance

    return 0