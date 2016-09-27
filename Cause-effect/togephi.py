"""
Export results into gephi link file
Author : Diviyan Kalainathan
Date : 26/07/2016
"""
import csv

inputfolder='output/obj8/pca_var/cluster_5/'
inputfile='results_lp_CSP+Public_thres0.12.csv'

output=open(inputfolder+'gephi_'+inputfile,'wb')
writer=csv.writer(output,delimiter=';',lineterminator='\n')
writer.writerow(['Source','Target','Score'])

with open(inputfolder+inputfile,'rb') as data:
    reader=csv.reader(data, delimiter=';')
    header=next(reader)
    for row in reader:
        if row[0][-4:]!='flag' and row[1][-4:]!='flag':
            if float(row[2])<0:
                #row.pop(2)
                row[0],row[1]=row[1],row[0]
            row[2]=str(abs(float(row[2])))
            writer.writerow(row)
output.close()
