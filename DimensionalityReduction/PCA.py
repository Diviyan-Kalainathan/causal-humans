'''
Principal component analysis on converted, filtered and prepared data
Author : Diviyan Kalainathan
Date : 1/06/2016

'''

import numpy, csv, re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# Load var info
num_bool = []
spec_note = []
type_var = []
color_type = []
category = []

vp = False  # Var to multiply W by sqrt of eigenvalue
mode = ''  # O for objective, S for subjective and '' to deactivate
obj_subj = []
nb_dimensions = 8

IDF = False
file_name = 'aa_sd_~8'

with open('input/Variables_info.csv', 'rb') as datafile:
    var_reader = csv.reader(datafile, delimiter=',')
    header_var = next(var_reader)
    for var_row in var_reader:
        type_var += [var_row[1]]
        num_bool += [var_row[3]]
        spec_note += [var_row[4]]
        category += [int(var_row[5])]
        obj_subj += [var_row[6]]

    category_type = []
    obj_subj_type = []
    row_len = 0
    for num_col in range(0, 541):
        if spec_note[num_col] != 'I':
            if type_var[num_col] == 'C' or (type_var[num_col] == 'D' and spec_note[num_col] == 'T'):
                row_len += 2  #
                color_type += ['C']
                color_type += ['FC']
                category_type += [category[num_col], category[num_col]]
                obj_subj_type += [obj_subj[num_col], [obj_subj[num_col]]]


            elif type_var[num_col] == 'D' and spec_note[num_col] != '-2' and spec_note[num_col] != 'T':
                # print(num_col)
                row_len += int(num_bool[num_col]) + 1
                for i in range(0, int(num_bool[num_col])):
                    color_type += ['D']
                    category_type += [category[num_col]]
                    obj_subj_type += [obj_subj[num_col]]

                color_type += ['FD']
                category_type += [category[num_col]]
                obj_subj_type += [obj_subj[num_col]]

print('Row len/n. features : ' + str(row_len))
# sarse_matrix = lil_matrix((input_length, row_len), dtype=numpy.int8)
# the matrix is now implemented in the csv result
# This computation is now made in order to cross-check the columns
# Load the dataset

n_features = row_len

print('--Load dataset--')
inputdata = numpy.zeros((n_features, 31112))
inc = 0
print len(category_type), len(color_type), row_len
with open('output/prepared_data.csv', 'rb') as datafile:
    var_reader = csv.reader(datafile, delimiter=';')
    header = next(var_reader)
    num_row = 0
    for row in var_reader:
        for i in range(0, n_features):
            if (not re.search('[a-zA-Z]', row[i])) and \
                    (mode == '' or (mode == 'O' and obj_subj_type[i] == 'O')
                     or (mode == 'S' and obj_subj_type[i] == 'S')):

                # if '.' not in row[i]:
                #    inputdata[i, num_row] = int(row[i])
                # else:
                inputdata[i, num_row] = float(row[i])
            else:
                inputdata[i, num_row] = 0.0
                inc += 1

        num_row += 1
        if num_row % 5000 == 0:
            print('.')
        elif num_row % 50 == 0:
            print('.'),

proc_data = inputdata
print('')
print(inc)
print('Done.')
# Transpose for the numpy.cov - computing the scatter matrix


## PCA ALGORITHM
print('---PCA ALGORITHM---')
# Centering & Normalization
print('--Centering and normalization--')
print(numpy.shape(proc_data))
for i in range(0, n_features):
    mean = numpy.mean(inputdata[i, :])

    sd = numpy.std(inputdata[i, :])
    if not IDF:
        if sd == 0:
            proc_data[i, :] = 0.0
        else:
            if sd < 1:
                print('SD < 1 value! : ' + str(sd))
            proc_data[i, :] = (inputdata[i, :] - mean) / (sd )  # The +1 may affect the results
    else:
        if color_type[i] == 'C':
            #Applying regular normalization
            if sd < 0.05:
                proc_data[i, :] = 0.0
            else:
                if sd < 1:
                    print('SD < 1 value! : ' + str(sd))
                proc_data[i, :] = (inputdata[i, :] - mean) / (sd)

        else:
            #Applying inverse document frequency smooth
            if not sum(inputdata[i, :]) == 0:
                print(sum(inputdata[i, :]),)
                proc_data[i, :] = inputdata[i, :] * numpy.log( 1 + 31112/sum(inputdata[i, :]))
            else:
                proc_data[i, :] = 0.0

print(proc_data)
print('Done.')
# print(inputdata[500])
# Scatter matrix
print('--Covariance Matrix--')
covmat = numpy.cov(proc_data)
print('Done.')
# Computing eigenvalues & eigenvectors
print('--Eigenvalues--')
eigvalues, eigvectors = numpy.linalg.eig(covmat)

# eigvalues_sorted.sort(reverse=True)
# print(eigvalues_sorted2)
#eigvalues=eigvalues.real
print(len(eigvalues))
print(numpy.shape(eigvalues))
sum2 = 0
func = []
for i in range(0, len(eigvalues)):
    sum2 += eigvalues[i] ** 2

for i in range(0, len(eigvalues)):
    sum1 = 0
    for j in range(0, i):
        sum1 += eigvalues[j] ** 2
    func += [sum1 / sum2]

for i in range(0, len(eigvalues)):
    plt.plot(i, func[i], 'go')

#plt.show()
plt.clf()

func2 = []
for i in range(0, len(eigvalues)):
    func2 += [eigvalues[i]]

for i in range(0, len(eigvalues)):
    plt.plot(i, func2[i], 'go')

#plt.show()
plt.clf()

s_eigvalues = numpy.sort(eigvalues)

print(s_eigvalues)

# Need to compute the value distributions according to the eigenvalues

print('Done.')

numpy.savetxt('output/eigvalues_' + file_name + '.csv', eigvalues, delimiter=';')

# Transform matrix
print('--Transform matrix & points--')

W = eigvectors[:, 0:nb_dimensions]
W = W.real
if vp:
    Vp_mat = numpy.eye(W.shape[1], W.shape[1])
    for i in range(W.shape[1]):
        print(eigvalues[i].real)
        Vp_mat[i, i] = numpy.sqrt(eigvalues[i].real)
    W = numpy.dot(W, Vp_mat)

print(numpy.shape(W))

W = numpy.transpose(W)

# Applying t-SNE to var projected in the n-dim subspace
model = TSNE(n_components=2)
numpy.set_printoptions(suppress=True)
toprint = model.fit_transform(numpy.transpose(W))

for i in range(0, len(toprint[:, 1])):
    if color_type[i] == 'C':
        plt.plot(toprint[i, 0], toprint[i, 1], 'bx')

    elif color_type[i] == 'FC':
        plt.plot(toprint[i, 0], toprint[i, 1], 'cx')

    elif color_type[i] == 'D':
        plt.plot(toprint[i, 0], toprint[i, 1], 'rx')

    elif color_type[i] == 'FD':
        plt.plot(toprint[i, 0], toprint[i, 1], 'yx')

plt.savefig('output/tsne1_'+file_name+'.pdf')
plt.clf()

categ_colors = ['0.75', 'g', 'r', 'b', 'y', 'c', 'm', 'k']

for i in range(0, len(toprint[:, 1])):
    plt.plot(toprint[i, 0], toprint[i, 1], 'x', color=categ_colors[category_type[i]])

plt.savefig('output/tsne2_'+file_name+'.pdf')
plt.clf()

# Color variables
toprint = eigvectors[:, 0:2]

for i in range(0, len(toprint[:, 1])):

    if color_type[i] == 'C':
        plt.plot(toprint[i, 0], toprint[i, 1], 'bx')

    elif color_type[i] == 'FC':
        plt.plot(toprint[i, 0], toprint[i, 1], 'cx')

    elif color_type[i] == 'D':
        plt.plot(toprint[i, 0], toprint[i, 1], 'rx')

    elif color_type[i] == 'FD':
        plt.plot(toprint[i, 0], toprint[i, 1], 'yx')

plt.savefig('output/2D_'+file_name+'.pdf')

# Transforming points into new subspace


print(numpy.shape(inputdata))
computed_data = numpy.dot(W, inputdata)
computed_data = computed_data.real

print('Done.')

W_t = numpy.transpose(W)

# Save data
print('--Save Data--')

with open('output/ident_vectors_' + file_name + '.csv', 'wb') as sortedfile:
    datawriter = csv.writer(sortedfile, delimiter=';', quotechar='|')
    datawriter.writerow([str(nb_dimensions) + ' dimensions eig-vectors'])

for i in range(0, n_features):
    output_s = []
    output_s += [header[i]]
    for j in range(0, nb_dimensions):
        output_s += [str(W_t[i, j])]
    with open('output/ident_vectors_' + file_name + '.csv', 'a') as sortedfile:
        datawriter = csv.writer(sortedfile, delimiter=';', quotechar='|',
                                lineterminator='\n')
        datawriter.writerow(output_s)

numpy.savetxt('output/computed_data_' + file_name + '.csv', computed_data, delimiter=';')
numpy.savetxt('output/raw_indent_vectors_' + file_name + '.csv', numpy.transpose(W), delimiter=';')
print('Done.')
