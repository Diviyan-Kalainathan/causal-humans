import csv
import pandas as pd
import os

def meta2target(pthfile):
    with open(os.path.dirname(pthfile)+'/Dtargets.csv','wb') as outputfile:
        inputdata=pd.read_csv(pthfile,header=None,sep='    |   |  | ')
        inputdata.columns=['name','target']
        w=csv.writer(outputfile)
        w.writerow(['SampleID','Target'])
        i=1
        for idx,row in inputdata.iterrows():
            if row['target']==1:
                w.writerow(['pairD'+str(i),1])
            elif row['target']==2:
                w.writerow(['pairD'+str(i),-1])
            else:
                w.writerow(['pairD'+str(i), 0])

            if i==26091:
                break
            i+=1



if __name__ == '__main__':
    pthfile='../CauseEffectDataset/pairmeta.txt'
    meta2target(pthfile)