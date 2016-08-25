'''
Principal component analysis on converted, filtered and prepared data, without flags
Author : Diviyan Kalainathan
Date : 9/06/2016
DEPRECATED

'''

import numpy, csv, re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load var info
type_var = []
num_bool = []
spec_note = []
color_type = []
real_color_type=[]

with open('input/Variables_info.csv', 'rb') as datafile:
    var_reader = csv.reader(datafile, delimiter=',')
    header_var = next(var_reader)
    for var_row in var_reader:
        type_var += [var_row[1]]
        num_bool += [var_row[3]]
        spec_note += [var_row[4]]

    row_len = 0
    real_row_len=0
    for num_col in range(0, 541):
        if spec_note[num_col] != 'I':
            if type_var[num_col] == 'C' or (type_var[num_col] == 'D' and spec_note[num_col] == 'T'):
                row_len += 1  #
                real_row_len+=2
                color_type += ['C']
                color_type += ['FC']
                real_color_type+=['C']
            elif type_var[num_col] == 'D' and spec_note[num_col] != '-2' and spec_note[num_col] != 'T':
                # print(num_col)
                row_len += int(num_bool[num_col])
                real_row_len +=int(num_bool[num_col])+1
                for i in range(0, int(num_bool[num_col])):
                    color_type += ['D']
                    real_color_type+=['D']
                color_type += ['FD']

print('Row len/n. features : ' + str(row_len))
# sarse_matrix = lil_matrix((input_length, row_len), dtype=numpy.int8)
# the matrix is now implemented in the csv result
# This computation is now made in order to cross-check the columns
# Load the dataset

n_features = row_len

print('--Load dataset--')
inputdata = numpy.zeros((n_features, 32693))

with open('output/prepared_data.csv', 'rb') as datafile:
    var_reader = csv.reader(datafile, delimiter=';')
    header = next(var_reader)
    num_column = 0
    for row in var_reader:
        num_row = 0
        for i in range(0, real_row_len):
            if color_type[i]=='C' or color_type[i]=='D':
                if not re.search('[a-zA-Z]', row[i]):
                    if '.' not in row[i]:
                        inputdata[num_row, num_column] = int(row[i])
                        num_row +=1
                    else:
                        inputdata[num_row, num_column] = float(row[i])
                        num_row +=1
                else:
                    inputdata[num_row, num_column] = 0
                    num_row +=1

        num_column += 1
        if num_column % 5000 == 0:
            print('.')
        elif num_column % 50 == 0:
            print('.'),

proc_data = inputdata
print('')
print('Done.')
# Transpose for the numpy.cov - computing the scatter matrix


## PCA ALGORITHM
print('---PCA ALGORITHM---')
# Centering & Normalization
print('--Centering and normalization--')
print(numpy.shape(proc_data))
for i in range(0, n_features):
    mean = numpy.mean(inputdata[i, :])

    variance = numpy.var(inputdata[i, :])

    if variance != 0:
        proc_data[i, :] = (inputdata[i, :] - mean) / variance
    else:
        proc_data[i, :] = 0
print(proc_data)
print('Done.')
# print(inputdata[500])
# Scatter matrix
print('--Scatter Matrix--')
covmat = numpy.cov(proc_data)
print('Done.')
# Computing eigenvalues & eigenvectors
print('--Eigenvalues--')
eigvalues, eigvectors = numpy.linalg.eig(covmat)

# eigvalues_sorted.sort(reverse=True)
# print(eigvalues_sorted2)

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

plt.show()

func2 = []
for i in range(0, len(eigvalues)):
    func2 += [eigvalues[i]]

for i in range(0, len(eigvalues)):
    plt.plot(i, func2[i], 'go')

plt.show()

s_eigvalues = numpy.sort(eigvalues)

print(s_eigvalues)

ind_val1 = 0  # numpy.where(eigvalues == s_eigvalues[n_features-1])

ind_val2 = 1  # numpy.where(eigvalues == s_eigvalues[n_features-2])

eig_vec1 = eigvectors[:, [ind_val1]]
eig_vec2 = eigvectors[:, [ind_val2]]

# Need to compute the value distributions according to the eigenvalues

print(eig_vec1)
print(eig_vec2)

print(ind_val1)
print(ind_val2)

print('Done.')
# print(repr(eig_vec1))
# print(s_eigvalues[782])
# print(eigvalues[0])
# print(eigvalues[1])
# print(repr(ind_val2))

# Transform matrix
# 5 Dimensions
print('--Transform matrix & points--')
# W = numpy.vstack((eig_vec1[:,0]/eigvalues[0],eig_vec2[:,0]/eigvalues[1],eigvectors[:,2]/eigvalues[2],
#                 eigvectors[:,3]/eigvalues[3],eigvectors[:,4]/eigvalues[4]))
W = numpy.vstack((eig_vec1[:, 0], eig_vec2[:, 0], eigvectors[:, 2],
                  eigvectors[:, 3], eigvectors[:, 4]))
W = W.real
print(numpy.shape(W))


model = TSNE(n_components=2)
numpy.set_printoptions(suppress=True)
toprint = model.fit_transform(numpy.transpose(W))

for i in range(0, len(toprint[:, 1])):
    if real_color_type[i] == 'C':
        plt.plot(toprint[i, 0], toprint[i, 1], 'bx')


    elif real_color_type[i] == 'D':
        plt.plot(toprint[i, 0], toprint[i, 1], 'rx')


plt.show()

# Color variables
toprint = (numpy.transpose(numpy.vstack((eig_vec1[:, 0] / eigvalues[0], eig_vec2[:, 0] / eigvalues[1]))))

for i in range(0, len(toprint[:, 1])):

    if real_color_type[i] == 'C':
        plt.plot(toprint[i, 0], toprint[i, 1], 'bx')

    elif real_color_type[i] == 'D':
        plt.plot(toprint[i, 0], toprint[i, 1], 'rx')


plt.show()
# Transforming points into new subspace


print(numpy.shape(inputdata))
computed_data = numpy.dot(W, inputdata)
computed_data = computed_data.real
'''for i in range (0,1000):
    plt.plot([computed_data[0, i]], [computed_data[1, i]],'bx')

plt.show()'''
print('Done.')
W_t = numpy.transpose(W)

with open('output/ident_vectors_5-wf.csv', 'wb') as sortedfile:
    datawriter = csv.writer(sortedfile, delimiter=';', quotechar='|')
    datawriter.writerow(['5 dimensions eig-vectors -without flags'])


final_header=[]
for i in range(0,real_row_len):
    if color_type[i]=='C' or color_type[i]=='D':
        final_header+=[header[i]]

for i in range(0, n_features):
    print(i)
    output_s = []
    output_s += [final_header[i]]
    for j in range(0, 5):
        output_s += [str(W_t[i, j])]
    with open('output/ident_vectors_5-wf.csv', 'a') as sortedfile:
        datawriter = csv.writer(sortedfile, delimiter=';', quotechar='|',
                                lineterminator='\n')
        datawriter.writerow(output_s)

# Save data
print('--Save Data--')
numpy.savetxt('output/computed_data5dim_2-wf.csv', computed_data, delimiter=';')
# numpy.savetxt('output/ident_vectors_5.csv', numpy.transpose(W), delimiter=';')

print('Done.')