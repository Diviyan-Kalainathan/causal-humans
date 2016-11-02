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
from itertools import cycle
from sklearn.metrics import auc

inputdata='../output/test/results/'

crit_names = ["Pearson's correlation",
              "Pvalue Pearson",
              "Chi2 test",
              "Mutual information",
              "Corrected Cramer's V",
              #"Lopez-Paz Causation coefficient",
              "FSIC",
              #"BF2d mutual info",
              #"BFMat mutual info"
              ]

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
           if row['Pairtype']=='4':
               Mprobes+=1

           FDR.append((N_max/N)*(Mprobes/Mprobes_max))

        results.append(FDR)


pickle.dump(results,open(inputdata+'res.p','wb'))
#results=pickle.load(open(inputdata+'res.p','rb'))

for i in range(len(results)-1):
    print(results[i]==results[-1])

print(len(results))
#print(results)
for i in results:
    plt.plot(range(len(i)),i)

plt.legend(crit_names,loc=4)
plt.xlabel('Number of probes retrieved')
plt.ylabel('False discovery rate')
plt.xscale('log')
plt.show()'''#FDR w/ probes

colors = cycle(['cyan', 'indigo', 'seagreen', 'gold', 'blue', 'darkorange','red','grey','darkviolet','mediumslateblue'])
for crit, color in zip (crit_names,colors):
    tpr=[]
    fpr=[]
    print(crit)
    try:
        with open("CEfinal_train_"+crit[:4]+'.csv','r') as results_file:
            df=pd.read_csv(results_file,sep=';')
            df=df.sort_values(by='Target', ascending=False)
       
            P=float((df['Pairtype']!=4).sum())
            N=float((df['Pairtype']==4).sum())
       
            TP=0.0
            FP=0.0
            for index,row in df.iterrows():
                if crit[:4]!='Lope':
                    if row['Pairtype']==4:
                        FP+=1.0
                    else:
                        TP+=1.0
                else:
                     if row['Pairtype']!=4:
                        FP+=1.0
                     else:
                        TP+=1.0
                tpr.append(TP/P) #TPR
                fpr.append(FP/N) #FPR
                    
            tpr,fpr= (list(t) for t in zip(*sorted(zip(tpr,fpr))))
            auc_score=auc(fpr,tpr)
            plt.plot(fpr,tpr,label=crit+' (area: '+str(auc_score)+')',color=color)
    except IOError:
        continue
        

plt.plot([0, 1], [0, 1], linestyle='--', color='k',
         label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve independancy criteria on Kaggle Data')
plt.legend(loc="lower right")
plt.show()
    
 

      
