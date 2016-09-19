"""
Sorting people into different files according to the clustering
Author : Diviyan Kalainathan
Date : 28/06/2016
"""
from lib_lopez_paz import experiment_challenge as lp
from multiprocessing import Process
import os,sys

inputdata = 'obj8'
lopez_paz = True
max_proc=int(sys.argv[1])
# Creating parameters
cluster_n = 1
jobs = []
while os.path.exists('output/' + inputdata + '/split_data/cluster_' + str(cluster_n)) :
    inputfilespath = []
    outputfilespath = []

    print('Cluster '+str(cluster_n))
    part_number = 0
    while os.path.exists('output/' + inputdata + '/split_data/cluster_' + str(cluster_n) + '/pairs_c_' + str(
            cluster_n) + '_p' + str(part_number) + '.csv'):
        inputfilespath.append('output/' + inputdata + '/split_data/cluster_' + str(cluster_n) + '/pairs_c_' + str(
            cluster_n) + '_p' + str(part_number) + '.csv')
        outputfilespath.append('output/' + inputdata + '/split_data/cluster_' + str(cluster_n) + '/results_lp_c_' + str(
            cluster_n) + '_p' + str(part_number) + '.csv')
        part_number += 1
    # Creating process
    lp.predict(inputfilespath,outputfilespath,max_proc)
    cluster_n += 1

print('End of program.')