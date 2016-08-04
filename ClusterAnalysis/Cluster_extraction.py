'''
Sorting people into different files according to the clustering
Author : Diviyan Kalainathan
Date : 28/06/2016
'''

import numpy, csv
from matplotlib import pyplot as plt
import Similarity_analysis

input_file = 'cluster_centers_c8_n500_r16-subj.csv'
output_folder = 'Cluster_separation_3'

####Preparing data analysis for V-test

with open('input/prepared_data.csv', 'rb') as datafile:
    var_reader = csv.reader(datafile, delimiter=';')
    header = next(var_reader)

# Load var info
num_bool = []
spec_note = []
type_var = []
color_type = []
category = []
obj_subj = []

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

####


print('Separating clusters on multiple files')
clustering_input = numpy.loadtxt('input/' + input_file, delimiter=';')
clusters = numpy.asarray(sorted(clustering_input, key=lambda x: x[1]))

name_clusters = (set((clusters[:, 0])))

'''with open('input/nc_filtered_data.csv', 'rb') as datafile:
    datareader = csv.reader(datafile, delimiter=';')
    header_input = next(datareader)

    # Create n output files for n clusters
    for i in name_clusters:
        with open('output/'+ output_folder +'/cluster_'+str(int(i))+'.csv', 'wb') as outputfile:
            datawriter = csv.writer(outputfile, delimiter=';', quotechar='|')
            datawriter.writerow(['Data extracted from : '+ input_file])
            datawriter.writerow(header_input)

    line_idx=0
    for row in datareader:
        with open('output/'+ output_folder +'/cluster_'+str(int(clusters[line_idx,0]))+'.csv', 'a') as outputfile:
            datawriter = csv.writer(outputfile, delimiter=';', quotechar='|',
                                    lineterminator='\n')
            datawriter.writerow(row)



        line_idx+=1
    print(line_idx)
    print(len(clusters[:,0]))
    Similarity_analysis.var_similarity(output_folder,len(name_clusters),len(header_input),header_input)'''  # No separation

print('V-test Analysis')
# Vtest analysis
prep_data = numpy.loadtxt('input/prep_numpyarray.csv', delimiter=';')
print('--Computing total mean and std values')
total_mean = numpy.zeros((prep_data.shape[0]))
total_std = numpy.zeros((prep_data.shape[0]))
num_total_items = float(prep_data.shape[1])
v_test_results = numpy.zeros((prep_data.shape[0], len(name_clusters)))

for ques in range(prep_data.shape[0]):
    total_mean[ques] = numpy.mean(prep_data[ques, :])
    total_std[ques] = numpy.std(prep_data[ques, :])
print('--Computing v-tests on all vars')
for var in range(row_len):
    print('--- var : ' + str(var))
    cluster_values = [[] for j in range(len(name_clusters))]

    for ppl in range(prep_data.shape[1]):
        cluster_values[int(clusters[ppl, 0])] += [prep_data[var, ppl]]

    for n_cluster in range(len(name_clusters)):
        res = 0
        if color_type[var] == 'C':
            try:

                res = ((numpy.mean(cluster_values[n_cluster]) - total_mean[var]) / numpy.sqrt(
                    ((num_total_items - (len(cluster_values[n_cluster])) / (num_total_items - 1)) * (
                        (numpy.power(total_std[var], 2)) / float(len(cluster_values[n_cluster]))))))


            except ZeroDivisionError:
                res = 0

        else:
            try:
                if numpy.sqrt(((num_total_items - len(cluster_values[n_cluster])) / (num_total_items - 1)) * (
                            1 - (numpy.sum(prep_data[var, :]) / num_total_items)) * (
                            (len(cluster_values[n_cluster]) * numpy.sum(prep_data[var, :])) / num_total_items)) < 0.0001:
                    raise ValueError

                res = ((numpy.sum(cluster_values[n_cluster]) - (
                    float(len(cluster_values[n_cluster])) * numpy.sum(prep_data[var, :])) / num_total_items) /
                       numpy.sqrt(((num_total_items - len(cluster_values[n_cluster])) / (num_total_items - 1)) * (
                           1 - (numpy.sum(prep_data[var, :]) / num_total_items)) * (
                                      (len(cluster_values[n_cluster]) * numpy.sum(prep_data[var, :])) / num_total_items)))


            except ZeroDivisionError and ValueError:
                res = 0
                print('ZDE')

        v_test_results[var, n_cluster] = res

with open('output/' + output_folder + '/v-tests.csv', 'wb') as outputfile:
    datawriter = csv.writer(outputfile, delimiter=';', quotechar='|')
    datawriter.writerow(['V-tests'])

    for var in range(row_len):
        w_row = []
        w_row += [header[var]]
        for n_cluster in range(len(name_clusters)):
            w_row += [str(v_test_results[var, n_cluster])]

        datawriter.writerow(w_row)

numpy.savetxt('output/'+output_folder+'/numpy-v-test.csv',v_test_results ,delimiter=';')

c=0
hist_data=numpy.zeros((len(name_clusters)))
for i in name_clusters:
    hist_data[c]= ((clusters[:, 0]).tolist).count(i)
    c+=1

plt.hist(hist_data)
plt.show()