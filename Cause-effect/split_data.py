"""
Split the input data
Author : Diviyan Kalainathan
Date : 28/06/2016
"""

import csv
import os
import sys
from multiprocessing import Process
import time

# Parameters
num_lines = 5000
#inputfolder = 'output/subj6/'
listfiles = ['pairs_c_', 'publicinfo_c_', 'predictions_c_']
heads=[['SampleID','A','B'],['SampleID','A type','B type'],['SampleID','Target']]
def splitfiles_cluster(inputfolder, cluster_n):
    num_head=0
    for filetype in listfiles:
        with open(inputfolder + 'cluster_' + str(cluster_n) + '/' + filetype + str(cluster_n) + '.csv') as pairs:
            datareader = csv.reader(pairs, delimiter=';', quotechar='|')
            header = next(datareader)
            #splitfilepath = ''
            #rows_towrite = [[]]

            sampleID=[]
            idx=0
            n_file = 0
            rows_towrite = [[]]
            splitfilepath = inputfolder + 'split_data/cluster_' + str(cluster_n) + '/' + filetype + str(
                cluster_n) + '_p' + str(n_file) + '.csv'
            with open(splitfilepath, 'wb') as splitfile:
                datawriter = csv.writer(splitfile, delimiter=',', quotechar='|', lineterminator='\n')
                datawriter.writerow(heads[num_head])

            for row in datareader:

                if (datareader.line_num ) % num_lines == 0:
                    # Dump lines in old file
                    if n_file >= 0:
                        with open(splitfilepath, 'a') as splitfile:
                            datawriter = csv.writer(splitfile, delimiter=',', quotechar='|', lineterminator='\n')
                            for line in rows_towrite:
                                if line !='':
                                    datawriter.writerow(line)
                        sys.stdout.write('--Cluster ' + str(cluster_n) + ' -- file ' + str(n_file) + '\n')
                        sys.stdout.flush()
                    # Switch files
                    n_file += 1
                    rows_towrite = [[]]
                    splitfilepath = inputfolder + 'split_data/cluster_' + str(cluster_n) + '/' + filetype + str(
                        cluster_n) + '_p' + str(n_file) + '.csv'
                    with open(splitfilepath, 'wb') as splitfile:
                        datawriter = csv.writer(splitfile, delimiter=',', quotechar='|', lineterminator='\n')
                        datawriter.writerow(heads[num_head])

                if row: # sort out blank lines
                    sampleID.append(row[0]) #Replace ID w/ number?
                    #row[0]='valid'+str(idx)
                    rows_towrite.append(row)
                    idx+=1

            # Dump lines for last file
            if n_file >= 0:
                with open(splitfilepath, 'a') as splitfile:
                    datawriter = csv.writer(splitfile, delimiter=',', quotechar='|', lineterminator='\n')
                    for line in rows_towrite:
                        datawriter.writerow(line)

                #Create index
                splitfilepath = inputfolder + 'split_data/cluster_' + str(cluster_n) + '/index_c_'+str(cluster_n) + '.csv'
                with open(splitfilepath, 'wb') as splitfile:
                    datawriter = csv.writer(splitfile, delimiter=',', quotechar='|', lineterminator='\n')
                    datawriter.writerow(['Idx','SampleID'])
                    for i in range(len(sampleID)):
                        datawriter.writerow(['valid'+str(i),sampleID[i]])
        num_head+=1

    #Remove second line
    print('--Cluster ' + str(cluster_n) +'-- Removing second line --')
    for filetype in listfiles:
        n_file = 0
        while os.path.exists(inputfolder + 'split_data/cluster_' + str(cluster_n) + '/' + filetype + str(
            cluster_n) + '_p' + str(n_file) + '.csv'):
            path=inputfolder + 'split_data/cluster_' + str(cluster_n) + '/' + filetype + str(
            cluster_n) + '_p' + str(n_file) + '.csv'
            with open(path, "r") as f:
                reader = list(csv.reader(f, delimiter=","))
                reader.pop(1)
                with open(path, "w") as out:
                    writer = csv.writer(out, delimiter=",")
                    for row in reader:
                        writer.writerow(row)
            n_file+=1
    sys.stdout.write('--Cluster ' + str(cluster_n) + ' -- Job finished !\n')
    sys.stdout.flush()

'''class file_splitter(threading.Thread):
    def __init__(self, inputfolder, cluster_n):
        super(file_splitter, self).__init__()
        self.indir = inputfolder
        self.num_cluster = cluster_n

    def run(self):
        splitfiles_cluster(self.indir, self.num_cluster)'''#No multithreading : multiprocessing instead


# Begin main
def split(inputfolder):
    cluster_n = 0
    if not os.path.exists(inputfolder + 'split_data'):
        os.makedirs(inputfolder + 'split_data')

    jobs = []
    while os.path.exists(inputfolder + 'cluster_' + str(cluster_n)) :
        if not os.path.exists(inputfolder + 'split_data/cluster_' +str(cluster_n)):
            os.makedirs(inputfolder + 'split_data/cluster_' + str(cluster_n))

        p= Process(target=splitfiles_cluster,args=(inputfolder,cluster_n,))
        p.start()
        jobs.append(p)
        cluster_n += 1
        time.sleep(0.5)

    time.sleep(5)
    for proc in jobs:
        proc.join()