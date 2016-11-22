"""
Convert Kaggle format data into 1 file by pair of variables
"""

import os
import numpy as np
import pandas as pd
import csv

def format_col(df):
    df.columns=['A','B']
    A=""
    B=""
    for idx,row in df.iterrows():
        A+=' '+str(float(row['A']))
        B+=' ' + str(float(row['B']))
    return A,B

def file_to_kag(pthtofile):
    SplID=[]
    type_info=[]
    i=1
    with open(pthtofile+'Dfinal_Pairs.csv','wb') as output:
        writer=csv.writer(output)
        writer.writerow(['SampleID','A','B'])

        while os.path.exists(pthtofile+str(i)+'.txt'):
            print(pthtofile+str(i)+'.txt')
            df=pd.read_csv(pthtofile+str(i)+'.txt',sep='    |   |  | ')
            A,B=format_col(df)
            SampleID= os.path.basename(pthtofile)+'D'+str(i)
            SplID.append(SampleID)
            type_info.append('Numerical')
            writer.writerow([SampleID,A,B])
            i+=1

    print(len(SplID))
    print(len(type_info))
    info=pd.DataFrame([SplID,type_info,type_info])
    info=info.transpose()
    info.columns=['SampleID','A type','B type']
    #info['SampleID']=pd.Series(SplID,index=info.index)
    #info['Type-A']=pd.Series(type_info,index=info.index)
    #info['Type-B']=pd.Series(type_info,index=info.index)
    info.to_csv(pthtofile+'Efinal_publicinfo.csv',index=False)
if __name__ == '__main__':

    inputfile_type='/home/diviyan/Documents/CausalPairwiseC/DLP_Generator/txtfiles/pair'
    file_to_kag(inputfile_type)