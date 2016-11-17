"""
Convert Kaggle format data into 1 file by pair of variables
"""

import os
import numpy as np
import pandas as pd
import csv

def format_col(df):
    df.columns=['A','B']
    A=[]
    B=[]
    for row in df.iterrows():
        A.append(' '+str(float(row[A])))
        B.append(' ' + str(float(row[B])))
    return A,B

def file_to_kag(pthtofile):

    i=0
    with open(pthtofile+'final.csv','rb') as output:
        writer=csv.writer(output)
        writer.writerow(['SampleID','A','B'])

        while os.path.exists(pthtofile+str(i)+'.txt'):
            df=pd.read_csv(pthtofile+str(i)+'.txt')
            A,B=format_col(df)
            SampleID= os.path.basename(pthtofile)+str(i)
            writer.writerow([SampleID,A,B])
            i+=1

if __name__ == '__main__':

    inputfile_type='~/Documents/MATLAB/sup3/train'
    file_to_kag(inputfile_type)