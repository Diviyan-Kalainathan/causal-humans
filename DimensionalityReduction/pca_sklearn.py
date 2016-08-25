'''
Applying PCA using the SKlearn library
Author : Diviyan Kalainathan
Date : 22/06/2016
'''

import numpy, csv, re
from sklearn.decomposition import PCA

# Load var info
type_var = []
num_bool = []
spec_note = []
color_type = []

with open('input/Variables_info_modif.csv', 'rb') as datafile:
    var_reader = csv.reader(datafile, delimiter=',')
    header_var = next(var_reader)
    for var_row in var_reader:
        type_var += [var_row[1]]
        num_bool += [var_row[3]]
        spec_note += [var_row[4]]

    row_len = 0
    for num_col in range(0, 541):
        if spec_note[num_col] != 'I':
            if type_var[num_col] == 'C' or (type_var[num_col] == 'D' and spec_note[num_col] == 'T'):
                row_len += 2  #
                color_type += ['C']
                color_type += ['FC']

            elif type_var[num_col] == 'D' and spec_note[num_col] == '-1':
                row_len += int(num_bool[num_col]) + 1
                for i in range(0, int(num_bool[num_col])):
                    color_type += ['I']
                color_type += ['I']

            elif type_var[num_col] == 'D' and spec_note[num_col] != '-2' and spec_note[num_col] != 'T':
                # print(num_col)
                row_len += int(num_bool[num_col]) + 1
                for i in range(0, int(num_bool[num_col])):
                    color_type += ['D']
                color_type += ['FD']




print('Row len/n. features : ' + str(row_len))
# sarse_matrix = lil_matrix((input_length, row_len), dtype=numpy.int8)
# the matrix is now implemented in the csv result
# This computation is now made in order to cross-check the columns
# Load the dataset

n_features = row_len

print('--Load dataset--')
inputdata = numpy.zeros((n_features, 31112))

with open('output/prepared_data.csv', 'rb') as datafile:
    var_reader = csv.reader(datafile, delimiter=';')
    header = next(var_reader)
    num_row = 0
    print(len(header))
    for row in var_reader:
        if num_row==0:
            print(numpy.shape(row))
        for i in range(0, n_features):
            if (not re.search('[a-zA-Z]', row[i]) ) and color_type[i]!='I':
                if '.' not in row[i]:
                    inputdata[i, num_row] = int(row[i])
                else:
                    inputdata[i, num_row] = float(row[i])
            else:
                inputdata[i, num_row] = 0
        num_row += 1
        if num_row % 5000 == 0:
            print('.')
        elif num_row % 50 == 0:
            print('.'),

print('')
print('Done.')

pca = PCA(n_components=10,whiten=False)

result= pca.fit_transform(inputdata)
numpy.savetxt('output/pca_result_2.csv',result,delimiter=';')
print(result)

print(pca.explained_variance_ratio_)
numpy.savetxt('output/pca_var_ratio_2.csv',pca.explained_variance_ratio_,delimiter=';')
numpy.savetxt('output/pca_var_2.csv',pca.explained_variance_,delimiter=';')

