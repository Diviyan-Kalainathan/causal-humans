# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:25:35 2016

@author: Philippe
"""
from __future__ import division
from __future__ import print_function

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn import preprocessing
from math import sqrt


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


np.random.seed( 1000 )

code = 'latin-1'

scaleCBS = pd.read_csv("scaleCBS.csv", sep=';',index_col = 0,  encoding = code)
avgCBS = pd.read_csv("avgCBS.csv", sep=';',index_col = 0,  encoding = code)

pca = PCA(n_components=9)

resultPCA = pca.fit_transform(scaleCBS)


X = preprocessing.scale(resultPCA)
#X = resultPCA




#range_n_clusters = [2, 3, 4, 5, 6]
range_n_clusters = [9]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.

#    clusterer = KMeans(n_clusters=n_clusters, random_state=10)


    clusterer = KMeans(n_clusters=n_clusters, n_init=100, init='k-means++', max_iter = 100)
    
    cluster_labels = clusterer.fit_predict(X)

    np.savetxt('cluster_labels', cluster_labels, delimiter=',') 
    
    
    scaleCBS['label'] = cluster_labels
    avgCBS['label'] = cluster_labels
    
    n = scaleCBS.shape[0]
    
    for k in range(0,n_clusters):
        samplesClasse = scaleCBS[scaleCBS.label == k] 
    
        samplesClasseRealValues = avgCBS[avgCBS.label == k]
        samplesClasseRealValues.to_csv('Cluster' + str(k) + '.csv', sep = ';', encoding = code )
        
        ng = samplesClasse.shape[0]

        meanVector = samplesClasse.mean(axis =0)
        
        indexPassVTest = []   
        
        for i in range(0,meanVector.shape[0]):
            
            VTest = abs(meanVector.iloc[i]/sqrt((n-ng)/(n-1)/ng))
            
            
            
            if(VTest > 3):
                indexPassVTest.append(i)
            
       
        VariableVTestCluster = meanVector[indexPassVTest].sort_values( axis=0, ascending=False)

        VariableVTestCluster.to_csv('VTestCluster' + str(k) + '.csv', sep = ';', encoding = code )
    
    

        
#     The silhouette_score gives the average value for all the samples.
#     This gives a perspective into the density and separation of the formed
#     clusters
        
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhoutte score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    




    # Labeling the clusters
    centers = clusterer.cluster_centers_
    
    # Draw white circles at cluster centers
    
    model = TSNE(n_components=2)
    np.set_printoptions(suppress=True)
    toprint = model.fit_transform(np.append(X, centers, axis=0))
    
    arrayToPrint = np.vsplit(toprint, np.array([X.shape[0]]))
    
    XToprint =  arrayToPrint[0]
    centersToprint = arrayToPrint[1] 
    

    
    ax2.scatter(XToprint[:, 0], XToprint[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)
    
#
    
    ax2.scatter(centersToprint[:, 0], centersToprint[:, 1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(centersToprint):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("TSNE visualization of the clustered data.")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()