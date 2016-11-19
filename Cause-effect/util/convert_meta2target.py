import csv
import pandas as pd
import os

def meta2target(pthfile):
    with open(os.path.dirname(pthfile)+'/E_Target.csv','wb') as outputfile:
        inputdata=pd.read_csv(pthfile,header=None)
        inputdata.columns=['name','target']
        w=csv.writer(outputfile)
        w.writerow(['SampleID','Target'])
        i=1
        for idx,row in inputdata.iterrows():
            if row['target']==1:
                w.writerow(['pairE'+str(i),1])
            elif row['target']==2:
                w.writerow(['pairE'+str(i),-1])
            else:
                w.writerow(['pairE'+str(i), 0])
            i+=1



if __name__ == '__main__':
    pthfile='/home/kaydhi/Documents/CausalPairwiseC/MPI_Generator/pairmeta.txt'
    meta2target(pthfile)