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
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
from matplotlib import pyplot as plt

BINARY = "Binary"
CATEGORICAL = "Categorical"
NUMERICAL = "Numerical"

path = "Dream5ChallengeData/"
test_set_name = 'net1_expression_data_InSilico'  # 'net3_expression_data_Ecoli_toy'#'net1_expression_data_InSilico'#

filenameData = test_set_name + ".tsv"

outputPath = "output/Dream5.2/"
outputCausalityPairs = test_set_name + "_pairs.csv"
outputPublicInfo = test_set_name + "_publicinfo.csv"
outputforCausation = test_set_name + "_fcausation.csv"
outputResults = test_set_name + "_results.tsv"
outputResultsContrib = test_set_name + "_results_contrib.tsv"
outputTarget = path + test_set_name + "_target.tsv"

df_input = pd.read_csv(path + filenameData, sep='\t', encoding="latin-1")
if len(sys.argv) < 2:
    print('Specify number of cores.')
    raise ValueError
nb_proc = int(sys.argv[1])
independancy_criterion = 0
deconvolution_method = 1
predict_method = 2
thresholdDepLink = 0.01
# epsilon=0.5
beta = 0.5
alpha = 0.1

"""
#0 : "Pearson's correlation"
#1 : "Abs Pearson's correlation",
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


var_names = df_input.columns.values
skel_mat = np.ones((len(var_names), len(var_names)))  # Skeleton matrix

#### Create Skeleton ####
print('Create skeleton'),

for idx1 in range(len(var_names) - 1):
    for idx2 in range(idx1 + 1, len(var_names)):
        var1 = var_names[idx1]
        var2 = var_names[idx2]
        skel_mat[idx1, idx2] = dc.dependency_functions[independancy_criterion](df_input[var1].values,
                                                                               df_input[var2].values, NUMERICAL,
                                                                               NUMERICAL)
        skel_mat[idx2, idx1] = skel_mat[idx1, idx2]

print(skel_mat)
# #Set diagonal terms
# for idx in range(len(var_names)):
#     skel_mat[idx,idx]=epsilon*1 #Reg. Hyperparameter?
print('...Done.')
#### Apply deconvolution ####
print('Deconvolution'),

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

    # Set diagonal terms to 0
    for idx in range(len(var_names)):
        skel_mat[idx, idx] = 0

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

    mat_new1 = np.dot(np.dot(U, D), np.linalg.inv(U))

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
    """Partial correlation coefficient"""
    inv_mat = np.linalg.inv(skel_mat)
    Gdir = np.zeros(inv_mat.shape)
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            if i != j:
                Gdir[i, j] = -inv_mat[i, j] / np.sqrt(inv_mat[i, i] * inv_mat[j, j])

else:
    raise ValueError
print('...Done.')

#### Causality computation ####

print('Causality computation')
# Prepare files
print('Prepare Files'),
df_output = pd.DataFrame(columns=["SampleID", "A", "B"], index=None)
df_publicinfo = pd.DataFrame(columns=["SampleID", "A type", "B type"], index=None)

print(Gdir)

for i in range(0, Gdir.shape[0]):

    for j in range(i + 1, Gdir.shape[1]):

        if (abs(Gdir[i, j]) > thresholdDepLink):
            a = df_input.iloc[:, i].values
            b = df_input.iloc[:, j].values

            aValuesParse = transformData(a)
            bValuesParse = transformData(b)

            sampleID = df_input.columns.values[i] + "-" + df_input.columns.values[j]

            newLignOutput = pd.DataFrame([[sampleID, aValuesParse, bValuesParse]],
                                         columns=["SampleID", "A", "B"])

            df_output = pd.concat([df_output, newLignOutput], ignore_index=True)

            newLignPublicinfo = pd.DataFrame([[sampleID, "Numerical", "Numerical"]],
                                             columns=["SampleID", "A type", "B type"])

            df_publicinfo = pd.concat([df_publicinfo, newLignPublicinfo], ignore_index=True)

print('Number of pairs : '),
print(len(df_output.index))
df_output.to_csv(outputPath + outputCausalityPairs, index=False, encoding='utf-8', sep=",")
df_publicinfo.to_csv(outputPath + outputPublicInfo, index=False, encoding='utf-8', sep=",")
print('...Done.')

# Compute causation
# Create output file if not exists
print('Compute causation'),
open(outputPath + outputforCausation, 'a').close()

cp.ce_pairs_predict(predict_method, [outputPath + outputCausalityPairs],
                    [outputPath + outputPublicInfo],
                    [outputPath + outputforCausation], nb_proc)
# Fetch results
df_causation_results = pd.read_csv(outputPath + outputforCausation)
df_causation_results.columns = ['SampleID', 'Value']
print('...Done.')

print('Output values')
results = []
# Write final results
for idx, row in df_causation_results.iterrows():
    v_names = row['SampleID'].split('-')
    if row['Value'] > 0:
        results.append([v_names[0], v_names[1], row['Value']])
    else:
        results.append([v_names[1], v_names[0], abs(row['Value'])])

df_results = pd.DataFrame(results)
df_results.columns = ['Source', 'Target', 'Score']
df_results = df_results.sort_values(by='Score', ascending=False)
df_results.to_csv(outputPath + outputResults, sep='\t', index=False)


################################################# Results ############################################################

#### Compare results to target values ####

df_target = pd.read_csv(outputTarget, sep='\t', encoding="latin-1")
df_target.columns = ['Source', 'Target', 'Score']

P = float(len(df_target.index))
N = float(len(var_names) * (len(var_names) - 1)) - P

TP = 0.0
FP = 0.0
tpr = []  # = recall
fpr = []
ppv = []

for idx, row in df_results.iterrows():
    if ((df_target['Source'] == row['Source']) & (df_target['Target'] == row['Target'])).any():  # Scores  are only 1 :
        # Need to modify if scores are different than 1.
        TP += 1
    else:
        FP += 1

    tpr.append(TP / P)  # TPR=recall
    fpr.append(FP / N)  # FPR
    ppv.append(TP / (TP + FP))

auc_pr_score = auc(tpr, ppv, reorder=True)
auc_roc_score = auc(fpr, tpr, reorder=True)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

pl1 = ax1.plot(fpr, tpr, label=' (area: {0:3f})'.format(auc_roc_score), color='r')
pl2 = ax2.plot(tpr, ppv, label=' (area: {0:3f})'.format(auc_pr_score), color='r')

ax1.plot([0, 1], [0, 1], linestyle='--', color='k',
         label='Luck')

ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve on directed graph')
ax1.legend(loc="lower right")
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision recall curve on directed graph')
ax2.legend(loc='best')
plt.show()
plt.savefig(outputPath + test_set_name + 'results_directed.pdf')
plt.clf()

#### Results on structure w/ causation coefficient ####

P = float(len(df_target.index))
N = float(len(var_names) * (len(var_names) - 1)) / 2 - P

TP = 0.0
FP = 0.0
tpr = []  # = recall
fpr = []
ppv = []

for idx, row in df_results.iterrows():
    if (((df_target['Source'] == row['Source']) & (df_target['Target'] == row['Target']))
            | (df_target['Source'] == row['Target']) & (
            df_target['Target'] == row['Source'])).any():  # Scores  are only 1 :
        # Need to modify if scores are different than 1.
        TP += 1
    else:
        FP += 1

    tpr.append(TP / P)  # TPR=recall
    fpr.append(FP / N)  # FPR
    ppv.append(TP / (TP + FP))

auc_pr_score = auc(tpr, ppv, reorder=True)
auc_roc_score = auc(fpr, tpr, reorder=True)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

pl1 = ax1.plot(fpr, tpr, label=' (area: {0:3f})'.format(auc_roc_score), color='r')
pl2 = ax2.plot(tpr, ppv, label=' (area: {0:3f})'.format(auc_pr_score), color='r')

ax1.plot([0, 1], [0, 1], linestyle='--', color='k',
         label='Luck')

ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve on graph structure')
ax1.legend(loc="lower right")
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision recall curve on graph structure')
ax2.legend(loc='best')
plt.show()
plt.savefig(outputPath + test_set_name + 'results_not_directed.pdf')
plt.clf()

#### Results on structure after deconvolution ####

P = float(len(df_target.index))
N = float(len(var_names) * (len(var_names) - 1)) / 2 - P

TP = 0.0
FP = 0.0
tpr = []  # = recall
fpr = []
ppv = []

for i in range(0, Gdir.shape[0]):
    for j in range(i + 1, Gdir.shape[1]):

        if (abs(Gdir[i, j]) > 0.001):
            if (((df_target['Source'] == var_names[i]) & (df_target['Target'] == var_names[j]))
                    | (df_target['Source'] == var_names[j]) & (
                            df_target['Target'] == var_names[i])).any():  # Scores  are only 1 :
                    # Need to modify if scores are different than 1.
                    TP += 1
            else:
                FP += 1

            tpr.append(TP / P)  # TPR=recall
            fpr.append(FP / N)  # FPR
            ppv.append(TP / (TP + FP))

auc_pr_score = auc(tpr, ppv, reorder=True)
auc_roc_score = auc(fpr, tpr, reorder=True)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

pl1 = ax1.plot(fpr, tpr, label=' (area: {0:3f})'.format(auc_roc_score), color='r')
pl2 = ax2.plot(tpr, ppv, label=' (area: {0:3f})'.format(auc_pr_score), color='r')

ax1.plot([0, 1], [0, 1], linestyle='--', color='k',
label='Luck')

ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve on graph structure')
ax1.legend(loc="lower right")
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision recall curve on graph structure')
ax2.legend(loc='best')
plt.show()
plt.savefig(outputPath + test_set_name + 'results_not_directed.pdf')
plt.clf()

#### Results on links with both contributions ####

idx1=0
idx2=1
contrib_results=[]
for idx, row in df_causation_results.iterrows():
    var=row['SampleID'].split('-')
    while ((var[0]!=var_names[idx1] | var[1]!=var_names[idx2])&(var[1]!=var_names[idx1] | var[0]!=var_names[idx2])):
        idx2+=1
        if idx2==len(var_names):
            idx1+=1
            idx2=idx1+1
            if idx1==len(var_names):
                print('Vars Not ordered')
                raise ValueError
    score= abs(Gdir[idx1,idx2])*row['Score']
    if score>0:
        contrib_results.append([var[0],var[1],score])
    else:
        contrib_results.append([var[1],var[0],score])

df_results_contrib=pd.DataFrame(contrib_results,columns=['Source','Target','Score'])
df_results_contrib=df_results_contrib.sort_values(by='Score', ascending=False)
df_results_contrib.to_csv(outputPath + outputResultsContrib, sep='\t', index=False)

P = float(len(df_target.index))
N = float(len(var_names) * (len(var_names) - 1)) - P

TP = 0.0
FP = 0.0
tpr = []  # = recall
fpr = []
ppv = []

for idx, row in df_results_contrib.iterrows():
    if ((df_target['Source'] == row['Source']) & (df_target['Target'] == row['Target'])).any():  # Scores  are only 1 :
        # Need to modify if scores are different than 1.
        TP += 1
    else:
        FP += 1

    tpr.append(TP / P)  # TPR=recall
    fpr.append(FP / N)  # FPR
    ppv.append(TP / (TP + FP))

auc_pr_score = auc(tpr, ppv, reorder=True)
auc_roc_score = auc(fpr, tpr, reorder=True)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

pl1 = ax1.plot(fpr, tpr, label=' (area: {0:3f})'.format(auc_roc_score), color='r')
pl2 = ax2.plot(tpr, ppv, label=' (area: {0:3f})'.format(auc_pr_score), color='r')

ax1.plot([0, 1], [0, 1], linestyle='--', color='k',
         label='Luck')

ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve on directed graph w/ both contributions')
ax1.legend(loc="lower right")
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision recall curve on directed graph w/ both contributions')
ax2.legend(loc='best')
plt.show()
plt.savefig(outputPath + test_set_name + 'results_directed_contrib.pdf')
plt.clf()
print('End of program.')
