"""
From causality-treated data; construct a graph of causality
Author : Diviyan Kalainathan
Date : 7/11/2016
"""

import csv
import cPickle as pkl
import numpy as np
import sys
import pandas as pd
import model_comparison.dependency_criterion as dc
import causal_predict as cp

BINARY = "Binary"
CATEGORICAL = "Categorical"
NUMERICAL = "Numerical"


inputfolder = 'output/obj8/pca_var/cluster_5/'
input_publicinfo = inputfolder+'publicinfo_c_5.csv'
causal_results = ''  # csv with 3 cols, Avar, Bvar & target
variables_data= inputfolder + 'pairs_c_5.csv'

path = "Dream5ChallengeData/"
filenameData = "net1_expression_data_InSilico.tsv"

outputPath = "output/Dream5/"

outputCausalityPairs = "net1_expression_data_InSilico_pairs.csv"
outputPublicInfo = "net1_expression_data_InSilico_publicinfo.csv"
outputTarget="net1_expression_data_InSilico_target.csv"



df_input = pd.read_csv(variables_data, sep='\t', encoding="latin-1")

independancy_criterion=0
deconvolution_method=1
predict_method=1
thresholdDepLink = 0.1


"""
#1 : "Pearson's correlation",
#2 : "Pval-Pearson",
#3 : "Chi2 test",
#4 : "Mutual information",
#5 : "Corrected Cramer's V",
#6 : "Lopez Paz's coefficient",
#7 : #"FSIC",
#8 : "BF2d mutual info",
#9 : "BFMat mutual info",
#10 : "ScPearson correlation",
#11 : "ScPval-Pearson"""


def transformData(x):
    transformedData = ""
    for values in x:
        transformedData += " " + str(float(values))

    return transformedData

var_names=df_input.columns.values
skel_mat=np.ones((len(var_names),len(var_names))) #Skeleton matrix

#### Create Skeleton ####

for idx1, var1 in enumerate(var_names[:-1]):
    for idx2, var2 in enumerate(var_names[idx1:]):
        skel_mat[idx1,idx2]=dc.crit_names[independancy_criterion](df_input[var1].values,df_input[var2].values,NUMERICAL,NUMERICAL)


#### Apply deconvolution ####
print('Deconvolution')

if deconvolution_method == 1:
    """This is a python implementation/translation of network deconvolution

 AUTHORS:
    Algorithm was programmed by Soheil Feizi.
    Paper authors are S. Feizi, D. Marbach,  M. Medard and M. Kellis

 REFERENCES:
   For more details, see the following paper:
    Network Deconvolution as a General Method to Distinguish
    Direct Dependencies over Networks
    By: Soheil Feizi, Daniel Marbach,  Muriel Medard and Manolis Kellis
    Nature Biotechnology"""  # Credits, Ref

    Gdir = np.dot(skel_mat, np.linalg.inv(np.identity(len(var_names)) + skel_mat))

elif deconvolution_method == 2:
    """This is a python implementation/translation of network deconvolution

    AUTHORS :
        B. Barzel, A.-L. Barab\'asi

    REFERENCES :
        Network link prediction by global silencing of indirect correlations
        By: Baruch Barzel, Albert-L\'aszl\'o Barab\'asi
        Nature Biotechnology"""  # Credits, Ref
    mat_diag = np.zeros((len(var_names), len(var_names)))
    D_temp = np.dot(skel_mat - np.identity(len(var_names)), skel_mat)
    for i in range(len(var_names)):
        mat_diag[i, i] = D_temp[i, i]
    Gdir = np.dot((skel_mat - np.identity(len(var_names)) + mat_diag), np.linalg.inv(skel_mat))

else:
    raise ValueError
print('Done.')


df_output = pd.DataFrame(columns=["SampleID", "A", "B"])
df_publicinfo = pd.DataFrame(columns=["SampleID", "A type", "B type"])


for i in range(0,Gdir.shape[0]):

    for j in range(i+1, Gdir.shape[1]):

        if(Gdir[i,j] > thresholdDepLink):


            a = df_input.iloc[:,i].values
            b = df_input.iloc[:,j].values

            aValuesParse = transformData(a)
            bValuesParse = transformData(b)

            sampleID = df_input.columns.values[i] + "-" + df_input.columns.values[j]

            newLignOutput = pd.DataFrame([[sampleID, aValuesParse, bValuesParse]],
                                         columns=["SampleID", "A", "B"])

            df_ouput = pd.concat([df_output, newLignOutput])

            newLignPublicinfo = pd.DataFrame([[sampleID, "Numerical", "Numerical"]],
                                         columns=["SampleID", "A type", "B type"])

            df_publicinfo = pd.concat([df_publicinfo, newLignPublicinfo])


df_output.to_csv(outputPath + outputCausalityPairs, index=False, encoding='utf-8', sep= ",")
df_publicinfo.to_csv(outputPath + outputPublicInfo, index=False, encoding='utf-8', sep= ",")

#Create file id not exists
open(outputPath+outputTarget,'a').close()

cp.ce_pairs_predict(predict_method,outputPath + outputCausalityPairs,outputPath + outputPublicInfo,outputPath+outputTarget,1)
