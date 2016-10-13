"""
Goal is to find the dependency criterion adapted to our data
Author : Diviyan Kalainathan
Date : 11/10/2016
"""

'''
The list of tested criterion are :
(#1 : Absolute value of Pearson's correlation)
#2 : Regular value of Pearson's correlation
#3 : Chi2 test
#4 : Mutual information
#5 : Corrected Cramer's V
#6 : Causation coefficient
#7 : HSIC
#8 : Lopez-Paz's dependency
(ICC?)
'''


import os,sys
import pandas as pd
import skeleton_construction_methods as scm
from multiprocessing import Pool
import scipy.stats as stats
from lib_fonollosa import features
import numpy
from sklearn import metrics
import csv


BINARY = "Binary"
CATEGORICAL = "Categorical"
NUMERICAL = "Numerical"

max_proc=int(sys.argv[1])
inputdata='../output/test/test_crit_'
crit_names=["Pearson's correlation",
            "Chi2 test",
            "Mutual information",
            "Corrected Cramer's V",
            "Causation coefficient"
            "HSIC"]




def confusion_mat(val1,val2):
    '''
    contingency_table = numpy.zeros((len(set(val1)), len(set(val2))))
    for i in range(len(val1)):
        contingency_table[list(set(val1)).index(val1[i]),
                          list(set(val2)).index(val2[i])] += 1'''
    contingency_table= numpy.asarray(pd.crosstab(numpy.asarray(val1,dtype='object'),numpy.asarray( val2, dtype='object')))
    # Checking and sorting out bad columns/rows
    max_len, axis_del = max(contingency_table.shape), [contingency_table.shape].index(
        max([contingency_table.shape]))
    toremove = [[], []]

    for i in range(contingency_table.shape[0]):
        for j in range(contingency_table.shape[1]):
            if contingency_table[i, j] < 4:  # Suppress the line
                toremove[0].append(i)
                toremove[1].append(j)
                continue

    for value in toremove:
        contingency_table = numpy.delete(contingency_table, value, axis=axis_del)

    return contingency_table

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """

    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return numpy.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def f_pearson(var1,var2,var1type,var2type):
    return abs(stats.pearsonr(var1, var2)[0])

def f_chi2_test(var1,var2,var1type,var2type,threshold_pval=0.05):
    values1, values2 = features.discretized_sequences(var1, var1type, var2, var2type)

    contingency_table = confusion_mat(values1, values2)

    if contingency_table.size > 0 and min(contingency_table.shape) > 1:
        chi2, pval, dof, expd = stats.chi2_contingency(contingency_table)
        if pval < threshold_pval:  # there is a link
            return  1
        else:
            return  0
    else:
        return  0


def f_mutual_info_score(var1,var2,var1type,var2type):

    values1, values2 = features.discretized_sequences(var1, var1type, var2, var2type)
    return  metrics.adjusted_mutual_info_score(values1, values2)

def f_corr_CramerV(var1,var2,var1type,var2type):

    values1, values2 = features.discretized_sequences(var1, var1type, var2, var2type)
    contingency_table = confusion_mat(values1, values2)

    if contingency_table.size > 0 and min(contingency_table.shape) > 1:
        return cramers_corrected_stat(contingency_table)
    else:
        return 0

def f_causation_c():
    return 0

def f_hsic():
    return 0

dependency_functions=[f_pearson,
                       f_chi2_test,
                       f_mutual_info_score,
                       f_corr_CramerV]

def process_job(part_number,index):
    data = pd.read_csv(inputdata+'p'+str(part_number)+'.csv', sep=';')
    #data.columns=['SampleID', 'A', 'B', 'A-Type', 'B-Type', 'Pairtype']
    results = []

    for idx, row in data.iterrows():
        var1 = row['A'].split()
        var2 = row['B'].split()
        var1 = [float(i) for i in var1]
        var2 = [float(i) for i in var2]

        res=dependency_functions[index](var1,var2,row['A-Type'],row['B-Type'])
        results.append([res,row['Pairtype']])

    r_df=pd.DataFrame(results,columns=['Target','Pairtype'])
    sys.stdout.write('Writing results for '+crit_names[index][:4]+'-'+str(part_number) +'\n')
    sys.stdout.flush()
    r_df.to_csv(inputdata + crit_names[index][:4]+'-'+str(part_number) + '.csv',sep=';',index=False)


for idx_crit,name in enumerate(crit_names):
    part_number=1
    print('OK')
    pool=Pool(processes=max_proc)
    while os.path.exists(inputdata+'p'+str(part_number)+'.csv'):
        print(part_number)
        pool.apply_async(process_job,args=(part_number,idx_crit))
        part_number+=1
    pool.close()
    pool.join()

    #Merging file
    with open(inputdata+crit_names[idx_crit][:4]+'.csv','wb') as mergefile:
        merger= csv.writer(mergefile,delimiter=';',lineterminator='\n')
        merger.writerow(['Target','Pairtype'])
        for i in range(1,part_number):
            with open(inputdata + crit_names[idx_crit][:4]+'-'+str(i) + '.csv','rb') as partfile:
                reader=csv.reader(partfile,delimiter=';')
                header=next(reader)
                for row in reader:
                    merger.writerow(row)
            os.remove(inputdata + crit_names[idx_crit][:4]+'-'+str(i) + '.csv')

    #chunksize=10**4



