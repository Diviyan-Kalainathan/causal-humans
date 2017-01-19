"""
Find the high contributing vars to PCA axis
Author : Diviyan Kalainathan
Date : 10/01/2017
"""

import pandas as pd

inputfile=pd.read_csv('output/ws~/ident_vectors_ws_vp~5.csv',index_col=0,sep="\t")
print(inputfile.columns)
print(inputfile)
# inputfile=inputfile.set_index(inputfile['cols'])
outputlist=[]
number_selected_vars= 3

for ncol in inputfile.columns :
    temp=inputfile[ncol]
    temp=temp.apply(lambda x:abs(x))
    temp=temp.sort_values(ascending=False)
    #print(temp)

    temp=temp.iloc[:number_selected_vars]
    templist=temp.index.tolist()
    print(templist)
    outputlist.append(templist)

outputpd=pd.DataFrame(outputlist)
outputpd.to_csv('output/ws-min-max-vars.csv',index=False)