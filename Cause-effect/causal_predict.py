"""
Sorting people into different files according to the clustering
Author : Diviyan Kalainathan
Date : 28/06/2016
"""

from lib_fonollosa import*
from lib_fonollosa import predict as fo

from lib_lopez_paz import experiment_challenge as lp


from multiprocessing import Process
import os,sys


if __name__=="__main__":

    # causal_predict_method = int(sys.argv[2])
    causal_predict_method = 2

    """Method used for causal prediction
    # 1 : Handcraft features J. Fonollosa method
    # 2 : RCC - Randomized Causation Coefficient D. Lopez-Paz et al. 2015 (Toward a learning theory of cause effect inference)
    # 3 : TODO : add Lopez-Paz method with neural network
    """

    inputdata = 'obj8'
    cluster_n = 1

    ###A decommenter pour passer en mode multi cluster
    # while os.path.exists('output/' + inputdata + '/split_data/cluster_' + str(cluster_n)) :
    #
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

    ###A commenter pour passer en mode multi cluster
    while cluster_n < 2:

        inputfilespath = []
        infopath = []
        outputfilespath = []

        
        # Benchmark test on Kaggle challenge data
        # SUP3 Test coming from Tuebingen pairs
        inputfilespath.append("datacauseeffect/CEpairs/SUP3/CEdata_train_pairs.csv")
        infopath.append("datacauseeffect/CEpairs/SUP3/CEdata_train_publicinfo.csv")
        if(causal_predict_method == 1):
            outputfilespath.append("output/benchmark/Fonollosa_SUP3.csv")
        elif(causal_predict_method == 2):
            outputfilespath.append("output/benchmark/LopezKernel_SUP3.csv")

        # SUP4 Test novel artificial data
        inputfilespath.append("datacauseeffect/CEpairs/SUP4/CEnovel_test_pairs.csv")
        infopath.append("datacauseeffect/CEpairs/SUP4/CEnovel_test_publicinfo.csv")
        if (causal_predict_method == 1):
            outputfilespath.append("output/benchmark/Fonollosa_SUP4.csv")
        elif (causal_predict_method == 2):
            outputfilespath.append("output/benchmark/LopezKernel_SUP4.csv")

        # Validation dataset (real and artificial data)
        inputfilespath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_pairs.csv")
        infopath.append("datacauseeffect/CEpairs/CEdata/CEfinal_valid_publicinfo.csv")
        if (causal_predict_method == 1):
            outputfilespath.append("output/benchmark/Fonollosa_validationset.csv")
        elif (causal_predict_method == 2):
            outputfilespath.append("output/benchmark/LopezKernel_validationset.csv")
        ########

        if(causal_predict_method == 1):

            modelPath = "lib_fonollosa/"

            for idx in range(len(inputfilespath)):

                fo.predict(inputfilespath[idx], infopath[idx], outputfilespath[idx], modelPath )


        elif(causal_predict_method == 2):


            lopez_paz = True
            # max_proc=int(sys.argv[1])
            max_proc = 3
            # Creating parameters

            modelPath = "lib_lopez_paz/"


            # Creating process
            lp.predict(inputfilespath,outputfilespath,modelPath, max_proc)

        cluster_n += 1

print('End of program.')