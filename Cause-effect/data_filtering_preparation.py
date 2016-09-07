"""
Filter out unwanted couples of variables and generate the final input file
Author : Diviyan Kalainathan
Date : 28/06/2016
"""

import numpy
import csv
import os
import threading
import time
import sys

inputfolder = 'output/subj6/'
var_info = 'input/Variables_info.csv'
verbose = False
max_threads = 7


lock_data = threading.Lock()



## Def functions & classes

def compare_t_task(var_1, c_data_header, cluster_data, inputfolder, cluster_n, verbose=False):
    print(' ')
    sys.stdout.write('----Var : '+str( c_data_header[var_1]) )
    sys.stdout.flush()
    print('Var : ', c_data_header[var_1], '  -'),
    outputrows_pairs=[[] for j in range(var_1 + 1, len(c_data_header))]
    outputrows_publicinfo=[[] for j in range(var_1 + 1, len(c_data_header))]
    outputrows_predict=[[] for j in range(var_1 + 1, len(c_data_header))]

    for var_2 in range(var_1 + 1, len(c_data_header)):
        idx=var_2-(var_1 + 1)
        # print('.'),
        if not (var_1 + 1 == var_2 and c_data_header[var_2][-4:] == 'flag'):  # Useless to consider
            # the case of a var and its flag

            data_to_load = [var_1, var_2]
            if verbose: print(c_data_header[var_1], '-', c_data_header[var_2]),
            # Taking flags into account when loading the data
            if c_data_header[var_1][-4:] == 'flag':
                var_1_flag = True
            else:
                var_1_flag = False
                data_to_load += [var_1 + 1]

            if c_data_header[var_2][-4:] == 'flag':
                var_2_flag = True
            else:
                var_2_flag = False
                data_to_load += [var_2 + 1]

            # Loading data
            data_to_load.sort()
            var_data = cluster_data[:, data_to_load]

            SampleID = c_data_header[var_1] + '-' + c_data_header[var_2]
            data_A = []
            data_B = []
            if verbose: print(data_to_load)
            # Load only data if there is no missing values

            for ppl in range(var_data.shape[0]):
                # print(var_data[ppl, 1],var_data[ppl, var_data.shape[1] - 1])
                # print((var_1_flag or int(var_data[ppl, 1]) == 1) and (
                #            var_2_flag or int(var_data[ppl, var_data.shape[1] - 1]) == 1))
                if (var_1_flag or int(var_data[ppl, 1]) == 1) and (
                            var_2_flag or int(var_data[ppl, var_data.shape[1] - 1]) == 1):
                    # if the data is valid
                    data_A.append(var_data[ppl, 0])

                    if var_1_flag:
                        data_B.append(var_data[ppl, 1])
                    else:
                        data_B.append(var_data[ppl, 2])

            if verbose: print(len(data_A))
            if len(data_A) > 25:

                # Convert data_A and data_B into the right shape
                A_values = ''
                B_values = ''
                for val in range(len(data_A)):
                    A_values = A_values + str(data_A[val]) + ' '
                    B_values = B_values + str(data_B[val]) + ' '

                # Type of A & B
                if type_var[var_1] == 'C':
                    typeA = 'Numerical'
                elif type_var[var_1] == 'D':
                    typeA = 'Categorical'
                elif type_var[var_1] == 'B':
                    typeA = 'Binary'
                else:
                    print('ERROR : No type for A at var : ', var_1, name_var[var_1])
                    typeA = ''

                if type_var[var_2] == 'C':
                    typeB = 'Numerical'
                elif type_var[var_2] == 'D':
                    typeB = 'Categorical'
                elif type_var[var_2] == 'B':
                    typeB = 'Binary'
                else:
                    print('ERROR : No type for B at var : ', var_2, name_var[var_2])
                    typeB = ''

                #stock values to access less data
                outputrows_pairs[idx]=[SampleID, A_values, B_values]
                outputrows_publicinfo[idx]=[SampleID, typeA, typeB]
                outputrows_predict[idx]=[SampleID, '0']

    # Updating pairs head
    lock_data.acquire()
    with open(inputfolder + 'cluster_' + str(cluster_n) + '/pairs_c_' + str(cluster_n) + '.csv',
              'a') as outputfile:
        datawriter = csv.writer(outputfile, delimiter=';', quotechar='|',
                                lineterminator='\n')
        for row in outputrows_pairs:
            datawriter.writerow(row)

    # Updating public info
    with open(inputfolder + 'cluster_' + str(cluster_n) + '/publicinfo_c_' + str(
            cluster_n) + '.csv',
              'a') as outputfile:
        datawriter = csv.writer(outputfile, delimiter=';', quotechar='|',
                                lineterminator='\n')
        for row in outputrows_publicinfo:
            datawriter.writerow(row)

    # Updating predictions head
    with open(inputfolder + 'cluster_' + str(cluster_n) + '/predictions_c_' + str(
            cluster_n) + '.csv',
              'a') as outputfile:
        datawriter = csv.writer(outputfile, delimiter=';', quotechar='|',
                                lineterminator='\n')
        for row in outputrows_predict:
            datawriter.writerow(row)
    lock_data.release()



class ActivePool(object):
    def __init__(self):
        super(ActivePool, self).__init__()
        self.active = []
        self.lock = threading.Lock()

    def makeActive(self, name):
        with self.lock:
            self.active.append(name)

    def makeInactive(self, name):
        with self.lock:
            self.active.remove(name)

    def numActive(self):
        with self.lock:
            return len(self.active)

    def __str__(self):
        with self.lock:
            return str(self.active)


class data_formatter(threading.Thread):
    def __init__(self, var_1, c_data_header, cluster_data, inputfolder, cluster_n, pool, verbose=False):
        super(data_formatter, self).__init__()
        self.name = threading.current_thread().name
        pool.makeActive(self.name)
        self.pool = pool
        self.num_var = var_1
        self.data_h = c_data_header
        self.c_data = cluster_data
        self.indir = inputfolder
        self.num_cluster = cluster_n
        self.verb = verbose

    def run(self):
        compare_t_task(self.num_var, self.data_h, self.c_data, self.indir, self.num_cluster)
        self.pool.makeInactive(self.name)


# Loading data info
print('--Loading data & parameters--')
with open('input/Variables_info.csv', 'rb') as datafile:
    var_reader = csv.reader(datafile, delimiter=',', quotechar='\'')
    header = next(var_reader)
    name_var = []
    type_var = []
    num_bool = []
    spec_note = []
    ignored = []
    for var_row in var_reader:  # Taking flags into account
        if not (var_row[0][-4:] == 'drap' or var_row[4] == '-1' or var_row[4] == 'I'):
            name_var += [var_row[0], var_row[0] + '_flag']
            type_var += [var_row[1], 'B']
            num_bool += [var_row[3], str(2)]
            spec_note += [var_row[4], str(-2)]

print(len(name_var))
# Generate files for all clusters
print('--Generating input files--')
cluster_n = 0
jobs = []
Threadpools = ActivePool()

while os.path.exists(inputfolder + 'cluster_' + str(cluster_n)):
    cluster_path = inputfolder + 'cluster_' + str(cluster_n) + '/data_c_' + str(cluster_n) + '.csv'

    print('--- Cluster n : ', cluster_n)
    # No filtering done

    # Open datafile to get the header
    with open(cluster_path, 'rb') as inputfile:
        datareader = csv.reader(inputfile, delimiter=';', quotechar='|')
        c_data_header = next(datareader)

    print(len(c_data_header))

    # Create output files
    with open(inputfolder + 'cluster_' + str(cluster_n) + '/pairs_c_' + str(cluster_n) + '.csv',
              'wb') as sortedfile:
        datawriter = csv.writer(sortedfile, delimiter=';', quotechar='|')
        datawriter.writerow(['SampleID', 'A', 'B'])

    with open(inputfolder + 'cluster_' + str(cluster_n) + '/publicinfo_c_' + str(cluster_n) + '.csv',
              'wb') as sortedfile:
        datawriter = csv.writer(sortedfile, delimiter=';', quotechar='|')
        datawriter.writerow(['SampleID', 'A type', 'B-type'])

    with open(inputfolder + 'cluster_' + str(cluster_n) + '/predictions_c_' + str(cluster_n) + '.csv',
              'wb') as sortedfile:
        datawriter = csv.writer(sortedfile, delimiter=';', quotechar='|')
        datawriter.writerow(['SampleID', 'Target'])

    # Load cluster data
    cluster_data = numpy.loadtxt(cluster_path, skiprows=1, delimiter=';')
    if verbose: print(cluster_data.shape)

    # Browse through vars


    for var_1 in range(len(c_data_header) - 1):
        job_assigned = False
        '''for var_2 in range(var_1 + 1, len(c_data_header)):
            print('.'),
            if not (var_1 + 1 == var_2 and c_data_header[var_2][-4:] == 'flag'):  # Useless to consider
                # the case of a var and its flag

                data_to_load = [var_1, var_2]
                if verbose: print(c_data_header[var_1],'-',c_data_header[var_2]),
                # Taking flags into account when loading the data
                if c_data_header[var_1][-4:] == 'flag':
                    var_1_flag = True
                else:
                    var_1_flag = False
                    data_to_load += [var_1 + 1]

                if c_data_header[var_2][-4:] == 'flag':
                    var_2_flag = True
                else:
                    var_2_flag = False
                    data_to_load += [var_2 + 1]

                #Loading data
                data_to_load.sort()
                var_data=cluster_data[:,data_to_load]


                SampleID = c_data_header[var_1] + '-' + c_data_header[var_2]
                data_A = []
                data_B = []
                if verbose: print(data_to_load)
                # Load only data if there is no missing values

                for ppl in range(var_data.shape[0]):
                    # print(var_data[ppl, 1],var_data[ppl, var_data.shape[1] - 1])
                    #print((var_1_flag or int(var_data[ppl, 1]) == 1) and (
                    #            var_2_flag or int(var_data[ppl, var_data.shape[1] - 1]) == 1))
                    if (var_1_flag or int(var_data[ppl, 1]) == 1) and (
                                var_2_flag or int(var_data[ppl, var_data.shape[1] - 1]) == 1):
                        # if the data is valid
                        data_A.append(var_data[ppl, 0])

                        if var_1_flag:
                            data_B.append(var_data[ppl, 1])
                        else:
                            data_B.append(var_data[ppl, 2])

                if verbose: print(len(data_A))
                if len(data_A) > 25:

                    # Convert data_A and data_B into the right shape
                    A_values = ''
                    B_values = ''
                    for val in range(len(data_A)):
                        A_values = A_values + str(data_A[val]) + ' '
                        B_values = B_values + str(data_B[val]) + ' '

                    #Type of A & B
                    if type_var[var_1] == 'C':
                        typeA = 'Numerical'
                    elif type_var[var_1] == 'D':
                        typeA = 'Categorical'
                    elif type_var[var_1] == 'B':
                        typeA = 'Binary'
                    else:
                        print('ERROR : No type for A at var : ', var_1, name_var[var_1])
                        typeA=''

                    if type_var[var_2] == 'C':
                        typeB = 'Numerical'
                    elif type_var[var_2] == 'D':
                        typeB = 'Categorical'
                    elif type_var[var_2] == 'B':
                        typeB = 'Binary'
                    else:
                        print('ERROR : No type for B at var : ', var_2, name_var[var_2])
                        typeB=''

                    # Updating pairs head
                    with open(inputfolder + 'cluster_' + str(cluster_n) + '/pairs_c_' + str(cluster_n) + '.csv',
                              'a') as outputfile:
                        datawriter = csv.writer(outputfile, delimiter=';', quotechar='|',
                                                lineterminator='\n')
                        datawriter.writerow([SampleID, A_values, B_values])

                    # Updating public info
                    with open(inputfolder + 'cluster_' + str(cluster_n) + '/publicinfo_c_' + str(
                            cluster_n) + '.csv',
                              'a') as outputfile:
                        datawriter = csv.writer(outputfile, delimiter=';', quotechar='|',
                                                lineterminator='\n')
                        datawriter.writerow([SampleID, typeA , typeB])

                    # Updating predictions head
                    with open(inputfolder + 'cluster_' + str(cluster_n) + '/predictions_c_' + str(
                            cluster_n) + '.csv',
                              'a') as outputfile:
                        datawriter = csv.writer(outputfile, delimiter=';', quotechar='|',
                                                lineterminator='\n')
                        datawriter.writerow([SampleID, '0'])'''  # If no multithreading
        while not job_assigned:
            if Threadpools.numActive() < max_threads:
                jobs.append(data_formatter(var_1, c_data_header, cluster_data, inputfolder, cluster_n, Threadpools))
                print(time.time(), ' - ', Threadpools.numActive())

                    #threading.Thread(target=data_formatter, name=str(time.ctime), args=(
                    #var_1, c_data_header, cluster_data, inputfolder, cluster_n, Threadpools)))
                jobs[len(jobs)-1].start()
                job_assigned = True

            time.sleep(3)

    cluster_n += 1  # Increment for the loop

for thread in jobs:
    thread.join()
print('End of program.')
