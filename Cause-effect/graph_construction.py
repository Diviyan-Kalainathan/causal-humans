"""
From causality-treated data; construct a graph of causality
Author : Diviyan Kalainathan
Date : 28/06/2016
"""

import csv
import cPickle as pkl
import numpy
import scipy.stats as stats
import sys
from lib_fonollosa import features
from sklearn import metrics

types = True  # If there is heterogenous data Need publicinfo file

inputfolder = 'output/obj8/pca_var/cluster_5/'
input_publicinfo = inputfolder+'publicinfo_c_5.csv'
causal_results = inputfolder + 'results_lp_CSP+Public_thres0.12.csv'  # csv with 3 cols, Avar, Bvar & target
epsilon_diag = 0.01

if 'obj8' in inputfolder:
    obj = True
else:
    obj = False

flags = False  # Taking account of flags

skeleton_construction_method = int(sys.argv[1])
""" Type of skeleton construction
#0X : Skip and load computed data made by method X
#1 : Absolute value of Pearson's correlation
#2 : Regular value of Pearson's correlation
#3 : Chi2 test
#4 : Mutual information
#5 : Causation coefficient
(#6 : HSIC?)
"""

if sys.argv[1][0] == '0':  # Choose which data to load w/ arg of type "01"
    load_skeleton = True
else:
    load_skeleton = False

deconvolution_method = int(sys.argv[2])
"""Method used for the deconvolution
#1 : Deconvolution according to Soheil Feizi
#2 : Recursive method according to Michele Sebag for causation coefficients
     #Issue about the meaning of the value of the causation coefficient
#3 : Deconvolution/global silencing by B. Barzel, A.-L. Barab\'asi
"""

print('Loading data')
ordered_var_names = pkl.load(open('input/header.p'))

if not flags:  # remove flag vars
    ordered_var_names = [x for x in ordered_var_names if 'flag' not in x]

if obj == True:
    num_axis = 8
else:
    num_axis = 5

for axis in range(num_axis):
    ordered_var_names.append('pca_axis_' + str(axis + 1))
    if flags:
        ordered_var_names.append('pca_axis_' + str(axis + 1) + '_flag')

link_mat = numpy.ones((len(ordered_var_names), len(ordered_var_names)))  # Matrix of links, fully connected
# 1 is linked and 0 unlinked,
BINARY = "Binary"
CATEGORICAL = "Categorical"
NUMERICAL = "Numerical"

print('Done.')

#### Pearson's correlation to remove links ####
print('Creating link skeleton')

if load_skeleton:
    print('Skipping construction & loading values')
    with open(inputfolder + 'link_mat_pval_' + str(skeleton_construction_method) + '.p', 'rb') as link_mat_file:
        link_mat = pkl.load(link_mat_file)

elif skeleton_construction_method < 5:

    with open(inputfolder + 'pairs_c_5.csv', 'rb') as pairs_file:
        datareader = csv.reader(pairs_file, delimiter=';')
        header = next(datareader)

        if types:
            typesfile = open(input_publicinfo, 'rb')
            typereader = csv.reader(typesfile, delimiter=';')
            type_header = next(typereader)

        threshold_pval = 0.05
        # threshold_pearsonc=0.5 #No threshold on correlation coefficient
        var_1 = 0
        var_2 = 0
        # Idea: go through the vars and unlink the skipped (not in the pairs file) pairs of vars.
        for row in datareader:
            try:
                types_row = next(typereader)
            except NameError:
                pass
            if row == []:  # Skipping blank lines
                continue

            pair = row[0].split('-')

            if not flags and ('flag' in pair[0] or 'flag' in pair[1]):
                continue  # Skipping values w/ flags

            # Finding the pair var_1 var_2 corresponding to the line
            # and un-linking skipped values
            while pair[0] != ordered_var_names[var_1]:
                if var_2 != len(ordered_var_names):
                    link_mat[var_1, var_2 + 1:] = 0
                var_1 += 1
                var_2 = 0

            skipped_value = False  # Mustn't erase checked values
            while pair[1] != ordered_var_names[var_2]:
                if skipped_value:
                    link_mat[var_1, var_2] = 0
                var_2 += 1
                skipped_value = True

            # Parsing values of table & removing artifacts
            var_1_value = [float(x) for x in row[1].split(' ') if x is not '']
            var_2_value = [float(x) for x in row[2].split(' ') if x is not '']

            if len(var_1_value) != len(var_2_value):
                raise ValueError

            if skeleton_construction_method < 3:
                if abs(stats.pearsonr(var_1_value, var_2_value)[1]) < threshold_pval:
                    if skeleton_construction_method == 1:
                        link_mat[var_1, var_2] = abs(stats.pearsonr(var_1_value, var_2_value)[0])
                    elif skeleton_construction_method == 2:
                        link_mat[var_1, var_2] = (stats.pearsonr(var_1_value, var_2_value)[0])
                else:
                    link_mat[var_1, var_2] = 0
            else:
                try:
                    var_1_type, var_2_type = types_row[1], types_row[2]

                except NameError:
                    var_1_type, var_2_type, = NUMERICAL, NUMERICAL

                values1, values2 = features.discretized_sequences(var_1_value, var_1_type, var_2_value, var_2_type)
                if skeleton_construction_method == 3:
                    contingency_table = numpy.zeros((len(set(values1)), len(set(values2))))
                    for i in range(len(values1)):
                        contingency_table[list(set(values1)).index(values1[i]),
                                          list(set(values2)).index(values2[i])] += 1

                    # Checking and sorting out bad columns/rows
                    max_len, axis_del = max(contingency_table.shape), [contingency_table.shape].index(
                        max([contingency_table.shape]))
                    toremove = [[], []]

                    for i in range(contingency_table.shape[0]):
                        for j in range(contingency_table.shape[1]):
                            if contingency_table[i, j] < 4:  # Suppress the line
                                toremove[0].append(i)
                                toremove[1].append(j)
                                continue

                    for value in toremove:
                        contingency_table = numpy.delete(contingency_table, value, axis=axis_del)

                    if contingency_table.size>0 and min(contingency_table.shape)>1:
                        chi2,pval,dof,expd=stats.chi2_contingency(contingency_table)
                        if pval<threshold_pval: #there is a link
                            link_mat[var_1, var_2] = 1
                        else:
                            link_mat[var_1, var_2] = 0

                    else:
                        link_mat[var_1, var_2] = 0

                elif skeleton_construction_method == 4:
                    link_mat[var_1, var_2] = metrics.adjusted_mutual_info_score(values1, values2)
    try:
        typesfile.close()
    except NameError:
        pass

    # Symmetrize matrix
    for col in range(0, (len(ordered_var_names) - 1)):
        for line in range(col + 1, (len(ordered_var_names))):
            link_mat[line, col] = link_mat[col, line]

    # Diagonal elts
    for diag in range(0, (len(ordered_var_names))):
        link_mat[diag, diag] = epsilon_diag #To guarantee non-singularity

#### Causality score to remove links ####

elif skeleton_construction_method == 5:

    with open(causal_results, 'rb') as pairs_file:
        datareader = csv.reader(pairs_file, delimiter=';')
        header = next(datareader)
        threshold = 0.12
        var_1 = 0
        var_2 = 0
        # Idea: go through the vars and unlink the skipped (not in the pairs file) pairs of vars.
        for row in datareader:

            if not flags and ('flag' in row[0] or 'flag' in row[1]):
                continue  # Skipping values w/ flags

            # Finding the pair var_1 var_2 corresponding to the line
            # and un-linking skipped values
            while row[0] != ordered_var_names[var_1]:
                if var_2 != len(ordered_var_names):
                    link_mat[var_1, var_2 + 1:] = 0
                var_1 += 1
                var_2 = 0

            skipped_value = False  # Mustn't erase checked values
            while row[1] != ordered_var_names[var_2]:
                if skipped_value:
                    link_mat[var_1, var_2] = 0
                var_2 += 1
                skipped_value = True

            if float(row[2]) > threshold:
                link_mat[var_1, var_2] = float(row[2])

    # Anti-symmetrize matrix
    for col in range(0, (len(ordered_var_names) - 1)):
        for line in range(col + 1, (len(ordered_var_names))):
            link_mat[line, col] = -link_mat[col, line]

    # Diagonal elts
    for diag in range(0, (len(ordered_var_names))):
        link_mat[diag, diag] = 0

else:
    raise ValueError
if skeleton_construction_method != 0:
    with open(inputfolder + 'link_mat_pval_' + str(skeleton_construction_method) + '.p', 'wb') as link_mat_file:
        pkl.dump(link_mat, link_mat_file)

print('Done.')

#### Loading causation data ####
# Go through all nodes and remove redundant links

list_var = []  # create blank list for name of vars
causality_links = []  # List of links between vars
# Init list var and causation links:
for name_var in ordered_var_names:
    list_var.append(name_var)
    causality_links.append([[], []])

# Import data, construction of data structure
with open(causal_results, 'rb') as inputfile:
    reader = csv.reader(inputfile, delimiter=';')
    header = next(reader)

    for row in reader:

        if (row[0]) not in list_var:
            list_var.append(row[0])
            causality_links.append([[], []])  # 0 for parents, 1 for children

        if (row[1]) not in list_var:
            list_var.append(row[1])
            causality_links.append([[], []])  # 0 for parents, 1 for children

        if float(row[2]) > 0:
            causality_links[list_var.index(row[0])][1].append(list_var.index(row[1]))
            causality_links[list_var.index(row[1])][0].append(list_var.index(row[0]))

        else:
            causality_links[list_var.index(row[0])][0].append(list_var.index(row[1]))
            causality_links[list_var.index(row[1])][1].append(list_var.index(row[0]))

with open(causal_results + 'causality.pkl', 'wb') as handle:
    pkl.dump(causality_links, handle)

with open(causal_results + 'list_vars.pkl', 'wb') as handle:
    pkl.dump(list_var, handle)

#### Apply deconvolution ####
print('Deconvolution')
if deconvolution_method == 1:
    """This is a python implementation/translation of network deconvolution

 AUTHORS:
    Algorithm was programmed by Soheil Feizi.
    Paper authors are S. Feizi, D. Marbach,  M. Medard and M. Kellis

 REFERENCES:
   For more details, see the following paper:
    Network Deconvolution as a General Method to Distinguish
    Direct Dependencies over Networks
    By: Soheil Feizi, Daniel Marbach,  Muriel Medard and Manolis Kellis
    Nature Biotechnology"""  # Credits, Ref

    Gdir = numpy.dot(link_mat, numpy.linalg.inv(numpy.identity(len(ordered_var_names)) + link_mat))

elif deconvolution_method == 2:

    # Creating all possible combinations:
    Gdir = link_mat
    causality_possibilites = [[[], []] for i in causality_links]
    # ToDO
    # 1. Generate all lists
    # 2. Generate up to n parents & n children

elif deconvolution_method == 3:
    """This is a python implementation/translation of network deconvolution

    AUTHORS :
        B. Barzel, A.-L. Barab\'asi

    REFERENCES :
        Network link prediction by global silencing of indirect correlations
        By: Baruch Barzel, Albert-L\'aszl\'o Barab\'asi
        Nature Biotechnology"""  # Credits, Ref
    mat_diag = numpy.zeros((len(ordered_var_names), len(ordered_var_names)))
    D_temp = numpy.dot(link_mat - numpy.identity(len(ordered_var_names)), link_mat)
    for i in range(len(ordered_var_names)):
        mat_diag[i, i] = D_temp[i, i]
    Gdir = numpy.dot((link_mat - numpy.identity(len(ordered_var_names)) + mat_diag), numpy.linalg.inv(link_mat))

else:
    raise ValueError
print('Done.')

#### Output values ####
print('Writing output files')
with open(inputfolder + 'deconv_links' + str(skeleton_construction_method) + str(deconvolution_method) + '.csv',
          'wb') as outputfile:
    writer = csv.writer(outputfile, delimiter=';', lineterminator='\n')
    writer.writerow(['Source', 'Target', 'Weight'])
    for var_1 in range(len(ordered_var_names) - 1):
        for var_2 in range(var_1 + 1, len(ordered_var_names)):
            if abs(Gdir[var_1, var_2]) > 0.001:  # ignore value if it's near 0
                # Find the causal direction
                if list_var.index(ordered_var_names[var_2]) in \
                        causality_links[list_var.index(ordered_var_names[var_1])][1]:
                    # var_2 is the child
                    writer.writerow([ordered_var_names[var_1], ordered_var_names[var_2], abs(Gdir[var_1, var_2])])
                elif list_var.index(ordered_var_names[var_2]) in \
                        causality_links[list_var.index(ordered_var_names[var_1])][0]:
                    # Var_2 is the parent
                    writer.writerow([ordered_var_names[var_2], ordered_var_names[var_1], abs(Gdir[var_1, var_2])])

print('Done.')
print('End of program.')
