'''
Principal component analysis on converted, filtered and prepared data, without some values
Author : Diviyan Kalainathan
Date : 10/06/2016

Note : Special rules are added to exclude some of the variables
'''

import numpy, csv, re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load var info
type_var = []
num_bool = []
spec_note = []
color_type = []
category = []
vp = False  # Var to multiply W by sqrt of eigenvalue
mode = ''  # O for objective, S for subjective and '' to deactivate
obj_subj=[]
nb_dimensions=6

IDF = True
file_name = 'ws_~6'

with open('input/Variables_info_modif.csv', 'rb') as datafile:
    var_reader = csv.reader(datafile, delimiter=',')
    header_var = next(var_reader)
    for var_row in var_reader:
        type_var += [var_row[1]]
        num_bool += [var_row[3]]
        spec_note += [var_row[4]]
        category += [int(var_row[5])]
        obj_subj+=[var_row[6]]

    row_len = 0
    category_type = []
    obj_subj_type=[]
    for num_col in range(0, 541):
        if spec_note[num_col] != 'I':
            if (type_var[num_col] == 'C' or (type_var[num_col] == 'D' and spec_note[num_col] == 'T') )and spec_note[num_col]!='-1':
                row_len += 2  #
                color_type += ['C']
                color_type += ['FC']
                category_type += [category[num_col], category[num_col]]
                obj_subj_type +=[obj_subj[num_col],obj_subj[num_col]]
            elif spec_note[num_col] == '-1':

                if type_var[num_col] == 'D' :
                    row_len += int(num_bool[num_col]) + 1
                    for i in range(0, int(num_bool[num_col])):
                        color_type += ['I']
                        category_type += [category[num_col]]
                        obj_subj_type += [obj_subj[num_col]]
                    color_type += ['I']
                    category_type += [category[num_col]]
                    obj_subj_type += [obj_subj[num_col]]
                else:
                    row_len += 2  #
                    color_type += ['I']
                    color_type += ['I']
                    category_type += [category[num_col], category[num_col]]
                    obj_subj_type += [obj_subj[num_col],[obj_subj[num_col]]]


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

with open('output/n_prepared_data.csv', 'rb') as datafile:
    var_reader = csv.reader(datafile, delimiter=';')
    header = next(var_reader)
    num_row = 0
    inc=0
    print(len(header))
    for row in var_reader:
        if num_row == 0:
            print(numpy.shape(row))
        for i in range(0, n_features):
            if (not re.search('[a-zA-Z]', row[i])) and color_type[i] != 'I' and \
                    (mode == '' or (mode == 'O' and obj_subj_type[i]=='O')
                     or (mode == 'S' and obj_subj_type[i]=='S')):

                if '.' not in row[i]:
                    inputdata[i, num_row] = int(row[i])
                else:
                    inputdata[i, num_row] = float(row[i])
            else:
                inputdata[i, num_row] =0
                inc+=1
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
        proc_data[i, :] = (inputdata[i, :] - mean) / (sd + 1)  # The +1 may affect the results
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
plt.clf()

func2 = []
for i in range(0, len(eigvalues)):
    func2 += [eigvalues[i]]

for i in range(0, len(eigvalues)):
    plt.plot(i, func2[i], 'go')
plt.title('Valeurs propres')
plt.xlabel('Numero de valeur propre')
plt.ylabel('Valeur')
plt.show()
plt.clf()

s_eigvalues = sorted(eigvalues, reverse=True)

print(s_eigvalues)
print(eigvalues)
numpy.savetxt('output/eigvalues_' + file_name + '.csv', eigvalues, delimiter=';')

print('Done.')
# print(repr(eig_vec1))
# print(s_eigvalues[782])
# print(eigvalues[0])
# print(eigvalues[1])
# print(repr(ind_val2))

# Transform matrix
# 5 Dimensions
print('--Transform matrix & points--')

# Taking the 8 first vectors
W = eigvectors[:, 0:nb_dimensions]
W = W.real
if vp:
    Vp_mat = numpy.eye(W.shape[1], W.shape[1])
    for i in range(W.shape[1]):
        Vp_mat[i, i] = numpy.sqrt(eigvalues[i].real)
    W = numpy.dot(W, Vp_mat)
print(numpy.shape(W))
W = numpy.transpose(W)

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
categ_colors=['0.75','g','r','b','y','c','m','k']

for i in range(0, len(toprint[:, 1])):

    plt.plot(toprint[i, 0], toprint[i, 1], 'x', color=categ_colors[category_type[i]])


plt.savefig('output/tsne2_'+file_name+'.pdf')
plt.clf()

# Color variables
toprint = (numpy.transpose(W))

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
'''for i in range (0,1000):
    plt.plot([computed_data[0, i]], [computed_data[1, i]],'bx')

plt.show()'''
print('Done.')
print('--Save Data--')
W_t = numpy.transpose(W)

with open('output/ident_vectors_' + file_name + '.csv', 'wb') as sortedfile:
    datawriter = csv.writer(sortedfile, delimiter=';', quotechar='|')
    datawriter.writerow(['10 dimensions eig-vectors -md'])

for i in range(0, n_features):
    output_s = []
    output_s += [header[i]]
    for j in range(0, nb_dimensions):
        output_s += [str(W_t[i, j])]
    with open('output/ident_vectors_' + file_name + '.csv', 'a') as sortedfile:
        datawriter = csv.writer(sortedfile, delimiter=';', quotechar='|',
                                lineterminator='\n')
        datawriter.writerow(output_s)

# Save data

numpy.savetxt('output/computed_data_' + file_name + '.csv', computed_data, delimiter=';')
numpy.savetxt('output/raw_indent_vectors_' + file_name + '.csv', numpy.transpose(W), delimiter=';')

print('Done.')
