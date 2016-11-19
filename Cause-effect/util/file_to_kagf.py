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
    for idx,row in df.iterrows():
        A.append(' '+str(float(row['A'])))
        B.append(' ' + str(float(row['B'])))
    return A,B

def file_to_kag(pthtofile):
    SplID=[]
    type_info=[]
    i=1
    #with open(pthtofile+'final_Pairs.csv','wb') as output:
        #writer=csv.writer(output)
        #writer.writerow(['SampleID','A','B'])

    while os.path.exists(pthtofile+str(i)+'.txt'):
        print(pthtofile+str(i)+'.txt')
        #df=pd.read_csv(pthtofile+str(i)+'.txt',sep=' ')
        #A,B=format_col(df)
        SampleID= os.path.basename(pthtofile)+'D'+str(i)
        SplID.append(SampleID)
        type_info.append('Numerical')
        #writer.writerow([SampleID,A,B])
        i+=1

    print(SplID)
    print(type_info)
    info=pd.DataFrame()
    info['SampleID']=SplID
    info['Type-A']=type_info
    info['Type-B']=type_info
    info.to_csv(pthtofile+'final_publicinfo.csv',index=False)
if __name__ == '__main__':

    inputfile_type='/home/kaydhi/Documents/CausalPairwiseC/DLP_Generator/pair'
    file_to_kag(inputfile_type)