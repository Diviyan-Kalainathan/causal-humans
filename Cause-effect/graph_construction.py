"""
From causality-treated data; construct a graph of causality
Author : Diviyan Kalainathan
Date : 28/06/2016
"""

import csv
import cPickle as pkl
import numpy
import scipy.stats as stats

inputfolder='output/obj8/pca_var/cluster_5/'
causal_results=inputfolder+'cluster_5/...' #csv with 3 cols, Avar, Bvar & target
if 'obj8' in inputfolder:
    obj=True
else:
    obj=False

mode = 1
"""
#1 : Constructing graph w/ Michele's method:
#2 : Isabelle's method/ "Classic" (Pearson's correlation + Deconvolution + orientation)
"""
if mode==1:

    list_var = []  # create blank list for name of vars
    causality_links = []  # List of links between vars

    # Import data, construction of data structure
    with open(causal_results,'rb') as inputfile:
        reader=csv.reader(inputfile, delimiter=';')
        header=next(reader)

        for row in reader:

            if (row[0]) not in list_var:
                list_var.append(row[0])
                causality_links.append([[],[]]) #0 for parents, 1 for children

            if (row[1]) not in list_var:
                list_var.append(row[1])
                causality_links.append([[],[]]) #0 for parents, 1 for children

            if float(row[3])>0:
                causality_links[list_var.index(row[0])][1].append(list_var.index(row[1]))
                causality_links[list_var.index(row[1])][0].append(list_var.index(row[0]))

            else:
                causality_links[list_var.index(row[0])][0].append(list_var.index(row[1]))
                causality_links[list_var.index(row[1])][1].append(list_var.index(row[0]))

    with open(causal_results+'causality.pkl', 'wb') as handle:
        pkl.dump(causality_links,handle)

    with open(causal_results+'list_vars.pkl', 'wb') as handle:
        pkl.dump(list_var,handle)


    #Creating all possible combinations:

    causality_possibilites=[[[],[]] for i in causality_links]
    #ToDO ASK MICHELE FOR THE DATA STRUCTURE . INFINITE POSSIBILITIES CAN BE GENERATED
     #1. Generate all lists
     #2. Generate up to n parents & n children

elif mode==2:
    ordered_var_names=pkl.load(open('input/header.p'))
    if obj==True:
        num_axis=8
    else:
        num_axis=5

    for axis in range(num_axis):
        ordered_var_names.append('pca_axis_'+str(axis+1))
        ordered_var_names.append('pca_axis_' + str(axis + 1)+'_flag')

    link_mat=numpy.ones((len(ordered_var_names),len(ordered_var_names))) #Matrix of links, fully connected
    #1 is linked and 0 unlinked,


    #### Pearson's correlation to remove links ####

    with open(inputfolder+'pairs_c_5.csv','rb') as pairs_file:
        datareader=csv.reader(pairs_file, delimiter=',')
        header=next(datareader)
        threshold=0.01
        var_1=0
        var_2=0
        #Idea: go through the vars and unlink the skipped (not in the pairs file) pairs of vars.
        for row in datareader:
            pair= row[0].split('-')

            #Finding the pair var_1 var_2 corresponding to the line
            # and un-linking skipped values
            while pair[0]!=ordered_var_names[var_1]:
                if var_2!=len(ordered_var_names):
                    link_mat[var_1,var_2+1:]=0
                var_1+=1
                var_2 = 0

            skipped_value=False #Mustn't erase checked values
            while pair[1] != ordered_var_names[var_2]:
                if skipped_value:
                    link_mat[var_1, var_2] = 0
                var_2 += 1
                skipped_value=True

            #Parsing values of table & removing artifacts
            var_1_value = [x for x in row[1].split(' ') if x is not '']
            var_2_value = [x for x in row[2].split(' ') if x is not '']

            if len(var_1_value)!=len(var_2_value):
                raise ValueError

            link_mat[var_1, var_2]=stats.pearsonr(var_1_value,var_2_value)[0]

    pkl.dump(link_mat,open(inputfolder+'link_mat_pval.p'))

    #### Apply deconvolution ####