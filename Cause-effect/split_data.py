"""
Split the input data
Author : Diviyan Kalainathan
Date : 28/06/2016
"""

import csv
import os
import sys
import threading
import time

# Parameters
num_lines = 5000
#inputfolder = 'output/subj6/'
listfiles = ['pairs_c_', 'predictions_c_', 'publicinfo_c_']


def splitfiles_cluster(inputfolder, cluster_n):
    for filetype in listfiles:
        with open(inputfolder + 'cluster_' + str(cluster_n) + '/' + filetype + str(cluster_n) + '.csv') as pairs:
            pairs_reader = csv.reader(pairs, delimiter=';')
            header = next(pairs_reader)
            splitfilepath = ''
            rows_towrite = []
            n_file = -1
            for row in pairs_reader:
                if (pairs_reader.line_num - 1) % num_lines == 0:
                    # Dump lines in old file
                    if n_file >= 0:
                        with open(splitfilepath, 'a') as splitfile:
                            datawriter = csv.writer(splitfile, delimiter=';', quotechar='|', lineterminator='\n')
                            for line in rows_towrite:
                                datawriter.writerow(line)
                        sys.stdout.write('--Cluster ' + str(cluster_n) + ' -- file ' + str(n_file)+'\n')
                        sys.stdout.flush()
                    # Switch files
                    n_file += 1
                    rows_towrite = []
                    splitfilepath = inputfolder + 'split_data/cluster_' + str(cluster_n) + '/' + filetype + str(
                        cluster_n) + '_p' + str(n_file) + '.csv'
                    with open(splitfilepath, 'wb') as splitfile:
                        datawriter = csv.writer(splitfile, delimiter=';', quotechar='|')
                        datawriter.writerow(header)

                rows_towrite += row
            # Dump lines for last file
            if n_file >= 0:
                with open(splitfilepath, 'a') as splitfile:
                    datawriter = csv.writer(splitfile, delimiter=';', quotechar='|', lineterminator='\n')
                    for line in rows_towrite:
                        datawriter.writerow(line)


class file_splitter(threading.Thread):
    def __init__(self, inputfolder, cluster_n):
        super(file_splitter, self).__init__()
        self.indir = inputfolder
        self.num_cluster = cluster_n

    def run(self):
        splitfiles_cluster(self.indir, self.num_cluster)


# Begin main
def split(inputfolder):
    cluster_n = 0
    if not os.path.exists(inputfolder + 'split_data'):
        os.makedirs(inputfolder + 'split_data')

    jobs = []
    while os.path.exists(inputfolder + 'cluster_' + str(cluster_n)):
        if not os.path.exists(inputfolder + 'split_data/cluster_' +str(cluster_n)):
            os.makedirs(inputfolder + 'split_data/cluster_' + str(cluster_n))

        jobs.append(file_splitter(inputfolder, cluster_n))
        jobs[len(jobs) - 1].start()
        cluster_n += 1
        time.sleep(0.5)

    time.sleep(5)
    for thread in jobs:
        thread.join()