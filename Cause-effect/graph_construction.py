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
outputResults="net1_expression_data_InSilico_results.csv"


df_input = pd.read_csv(variables_data, sep='\t', encoding="latin-1")

nb_proc=int(sys.argv[1])
independancy_criterion=0
deconvolution_method=1
predict_method=1
thresholdDepLink = 0.1
# epsilon=0.5
beta = 0.5
alpha = 0.1

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
        skel_mat[idx2,idx1]=skel_mat[idx1,idx2]

# #Set diagonal terms
# for idx in range(len(var_names)):
#     skel_mat[idx,idx]=epsilon*1 #Reg. Hyperparameter?

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

    # Gdir = np.dot(skel_mat, np.linalg.inv(np.identity(len(var_names)) + skel_mat))

    """Author code transposed from matlab to python"""

    # pre - processing the input matrix
    # mapping between 0 and 1
    skel_mat = (skel_mat - np.min(skel_mat)) / (np.max(skel_mat) - np.min(skel_mat))

    #Set diagonal terms to 0
    for idx in range(len(var_names)):
        skel_mat[idx,idx] = 0

    # thresholding the input matrix
    y = np.percentile(skel_mat, alpha * 100)
    skel_mat[skel_mat < y] = 0


    D, U = np.linalg.eig(skel_mat)

    lam_n = abs(min(np.min(D), 0))
    lam_p = abs(max(np.max(D), 0))


    m1 = lam_p * (1 - beta) / beta;
    m2 = lam_n * (1 + beta) / beta;
    m = max(m1, m2);


    D = D * np.identity(D.shape[0])

    for i in range(0, D.shape[0]):
        D[i, i] = D[i, i] / (m + D[i, i]);

    mat_new1 = U * D * np.linalg.inv(U)

    m2 = np.min(mat_new1);
    mat_new2 = (mat_new1 + max(-m2, 0));

    m1 = np.min(mat_new2);
    m2 = np.max(mat_new2);

    Gdir = (mat_new2 - m1) / (m2 - m1);



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

elif deconvolution_method == 3:
    inv_mat=np.inv(skel_mat)
    Gdir=np.zeros(inv_mat.shape)
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            if i!=j:
                Gdir[i,j]=-inv_mat[i,j]/np.sqrt(inv_mat[i,i]*inv_mat[j,j])

else:
    raise ValueError
print('Done.')

#### Causality computation ####

#Prepare files
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

            df_output = pd.concat([df_output, newLignOutput])

            newLignPublicinfo = pd.DataFrame([[sampleID, "Numerical", "Numerical"]],
                                         columns=["SampleID", "A type", "B type"])

            df_publicinfo = pd.concat([df_publicinfo, newLignPublicinfo])


df_output.to_csv(outputPath + outputCausalityPairs, index=False, encoding='utf-8', sep= ",")
df_publicinfo.to_csv(outputPath + outputPublicInfo, index=False, encoding='utf-8', sep= ",")

#Compute causation
#Create output file if not exists
open(outputPath+outputTarget,'a').close()

cp.ce_pairs_predict(predict_method,outputPath + outputCausalityPairs,outputPath + outputPublicInfo,outputPath+outputTarget,nb_proc)
#Fetch results
df_causation_results=pd.read_csv(outputPath+outputTarget,columns=['SampleID','Value'])

results=[]
#Write final results
for idx,row in df_causation_results.iterrows():
    v_names=row['SampleID'].split('-')
    if row['Value']>0:
        results.append([v_names[0],v_names[1],row['Value']])
    else:
        results.append([v_names[1],v_names[0],abs(row['Value'])])

df_results=pd.DataFrame(results,columns=['Source','Target','Score'])
df_results.to_csv(outputPath+outputResults,sep='\t', index=False)