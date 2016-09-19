"""
Merge the results
Author : Diviyan Kalainathan
Date : 28/06/2016
"""

import os
import csv

idxed_files = True  # If the files are indexed

cluster_n = 0
inputdata = 'obj8'
threshold = 0.15

if inputdata == 'subj6':
    legend = ['RAS', 'Stress', 'Indep', 'Heur', 'Malh', 'Chgts']
else:
    legend = ['Indep', 'Sante', 'Ouvriers', 'CSP+Prive', 'ServPart', 'CSP+Public', 'Immigr', 'Accid']

while os.path.exists('output/' + inputdata + '/split_data/cluster_' + str(cluster_n)):
    part_number = 0
    # dic=[[],[]]
    # Creating output file
    outputfile = open(
        'output/' + inputdata + '/cluster_' + str(cluster_n) + '/results_lp_' + legend[cluster_n] + '_thres' + str(
            threshold) + '.csv',
        'wb')
    datawriter = csv.writer(outputfile, delimiter=';', quotechar='|', lineterminator='\n')
    datawriter.writerow(['SampleID - ' + legend[cluster_n], 'Target'])

    # Loading idx
    if idxed_files:
        with open('output/' + inputdata + '/split_data/cluster_' + str(cluster_n) + '/index_c_' + str(
                cluster_n) + '.csv', 'rb') as idxfile:
            reader = csv.reader(idxfile, delimiter=',')
            header = next(reader)
            # Create a dictionnary
            dic_varnames = dict([(row[0], row[1]) for row in reader])

    while os.path.exists('output/' + inputdata + '/split_data/cluster_' + str(cluster_n) + '/results_lp_c_' + str(
            cluster_n) + '_p' + str(part_number) + '.csv'):
        with open('output/' + inputdata + '/split_data/cluster_' + str(cluster_n) + '/results_lp_c_' + str(
                cluster_n) + '_p' + str(part_number) + '.csv', 'rb') as res_part:
            readertocopy = csv.reader(res_part, delimiter=';')
            d_header = next(readertocopy)
            for r_row in readertocopy:
                if abs(float(r_row[1])) > threshold:
                    if idxed_files:
                        r_row[0] = dic_varnames[r_row[0]]
                    datawriter.writerow(r_row)
        part_number += 1

    outputfile.close()
    cluster_n += 1
