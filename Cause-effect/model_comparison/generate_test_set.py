"""
Generation of the test set, w/ permuted vals
Author : Diviyan Kalainathan
Date : 11/10/2016
"""
import os
import pandas as pd
from random import randint
from random import shuffle
import csv
from multiprocessing import Pool
import numpy

inputdata='../output/obj8/pca_var/cluster_5/pairs_c_5.csv'
outputfolder='../output/test/'

max_gen=10
if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)

outputfile= open(outputfolder + 'test_crit.csv','wb')#Create file
writer=csv.writer(outputfile,lineterminator='\n')
writer.writerow(['SampleID','A','B','Pairtype'])
outputfile.close()


chunksize=10**4
data=pd.read_csv(inputdata,sep=';', chunksize=chunksize)
print(chunksize)
for chunk in data:

    chunk.dropna(how='all')
    chunk['Pairtype']='O'

    #Drop flags:
    chunk=chunk[chunk.SampleID.str.contains('flag')]
    to_add = []

    for index, row in chunk.iterrows():
        var1= row['A']
        var2= row['B']

        for i in range(max_gen):
            mode=randint(1,3)
            if mode == 1:
                var1=var1.split()
                shuffle(var1)
                var1=" ".join(str(j) for j in var1)
            elif mode == 2:
                var2 = var2.split()
                shuffle(var2)
                var2 = " ".join(str(j) for j in var2)
            elif mode == 3:
                var1 = var1.split()
                shuffle(var1)
                var1 = " ".join(str(j) for j in var1)
                var2 = var2.split()
                shuffle(var2)
                var2 = " ".join(str(j) for j in var2)

            to_add.append([row['SampleID']+'_Probe'+str(i),var1,var2,'P'])

    df2=pd.DataFrame(to_add,columns=['SampleID','A','B','Pairtype'])
    chunk.append(df2)
    #chunk= chunk.iloc[numpy.random.permutation(len(chunk))] #No need to shuffle
    #chunk.reset_index(drop=True)
    chunk.to_csv(outputfolder + 'test_crit.csv',mode='a',header=False,sep=';')
