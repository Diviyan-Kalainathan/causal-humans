"""
Sorting people into different files according to the clustering
Author : Diviyan Kalainathan
Date : 28/06/2016
"""
from lib_lopez_paz import experiment_challenge as lp
from multiprocessing import Process
import os,sys


if __name__=="__main__":


    causal_predict_method = int(sys.argv[1])
    """Method used for the deconvolution
    # 1 : J. Fonolossa
    # 2 : Randomized Causation Coefficient D. Lopez-Paz
    # 3 : Deconvolution/global silencing by B. Barzel, A.-L. Barab\'asi


    inputdata = 'obj8'
    lopez_paz = True
    # max_proc=int(sys.argv[1])
    max_proc = 1
    # Creating parameters
    cluster_n = 1
    jobs = []
    modelPath = "lib_lopez_paz/"

    # while os.path.exists('output/' + inputdata + '/split_data/cluster_' + str(cluster_n)) :
    #     inputfilespath = []
    #     outputfilespath = []
    #
    #     print('Cluster '+str(cluster_n))
    #     part_number = 0
    #     while os.path.exists('output/' + inputdata + '/split_data/cluster_' + str(cluster_n) + '/pairs_c_' + str(
    #             cluster_n) + '_p' + str(part_number) + '.csv'):
    #         inputfilespath.append('output/' + inputdata + '/split_data/cluster_' + str(cluster_n) + '/pairs_c_' + str(
    #             cluster_n) + '_p' + str(part_number) + '.csv')
    #         outputfilespath.append('output/' + inputdata + '/split_data/cluster_' + str(cluster_n) + '/results_lp_c_' + str(
    #             cluster_n) + '_p' + str(part_number) + '.csv')
    #         part_number += 1
    #
    #
    #
    #     # Creating process
    #     lp.predict(inputfilespath,outputfilespath,max_proc)
    #     cluster_n += 1



    inputfilespath = []
    outputfilespath = []
    # Benchmark test on SUP3 Tuebingen pairs
    inputfilespath.append("datacauseeffect/CEpairs/SUP3/CEdata_train_pairs.csv")
    # Benchmark test on validation test
    # inputfilespath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_pairs.csv")

    outputfilespath.append("output/benchmark/LopezKernel_SUP3.csv")

    lp.predict(inputfilespath,outputfilespath,modelPath, max_proc)



print('End of program.')