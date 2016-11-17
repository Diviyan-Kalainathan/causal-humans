"""
Convert Kaggle format data into 1 file by pair of variables
"""

import os
import numpy as np
import pandas as pd

def kag_to_file(pthtofile):
    dirname=os.path.dirname(pthtofile)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    inputdata=pd.read_csv(pthtofile)
    inputdata.columns=['SampleID','A','B']
    if 'pairs' in pthtofile:
        for idx, row in inputdata.iterrows():
            A=row['A'].split(' ')
            B=row['B'].split(' ')
            A.pop(0)
            B.pop(0)
            A=[float(i) for i in A]
            B=[float(i) for i in B]
            dfoutput=pd.DataFrame()
            dfoutput['A']=A
            dfoutput['B']=B
            dfoutput.to_csv(dirname+'/'+row['SampleID']+'.txt',sep=',',columns=None,header=False,index=False)
            print(row['SampleID'])

if __name__ == '__main__':

    inputfile='~/Documents/MATLAB/sup3/CEdata_train_pairs.csv'
    kag_to_file(inputfile)