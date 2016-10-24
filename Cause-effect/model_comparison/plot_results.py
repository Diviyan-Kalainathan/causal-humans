"""
Plot results of the criterion
Author : Diviyan Kalainathan
Date : 11/10/2016
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy

inputdata='../output/test/results/'

crit_names = ["Pearson's correlation",
              "Chi2 test",
              "Mutual information",
              "Corrected Cramer's V",
              "Causation coefficient",
              "FSIC",
              "BF2d mutual info",
              "BFMat mutual info"]

results=[]

'''for crit in crit_names:
    inputfile=inputdata+'test_crit_'+crit[:4]+'.csv'
    if os.path.exists(inputfile):
        print(crit)
        df=pd.read_csv(inputfile,sep=';')
        df=df[numpy.isfinite(df['Target'])]
        #df['Target']=df['Target'].astype('float')
        print(df.dtypes)
        df=df.sort_values(by='Target', ascending=False)
        print(df)
        N_max=len(df.index)
        print(N_max)
        N=0.0
        Mprobes_max = (df['Pairtype']=='P').sum()
        print(Mprobes_max)
        Mprobes = 0.0
        FDR=[]

        for index,row in df.iterrows():
           N=N+1
           if row['Pairtype']=='P':
               Mprobes+=1

           FDR.append((N_max/N)*(Mprobes/Mprobes_max))

        results.append(FDR)


pickle.dump(results,open(inputdata+'res.p','wb'))'''
results=pickle.load(open(inputdata+'res.p','rb'))
#
for i in range(len(results)-1):
    print(results[i]==results[-1])

print(len(results))
#print(results)
for i in results:
    plt.plot(range(len(i)),i)

plt.legend(crit_names,loc=4)
plt.xlabel('Number of probes retrieved')
plt.ylabel('False discovery rate')
plt.show()
