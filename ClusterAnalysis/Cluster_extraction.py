'''
Sorting people into different files according to the clustering
Author : Diviyan Kalainathan
Date : 28/06/2016
'''

import numpy,csv
import Similarity_analysis

input_file='cluster_predictions_c5_n500_r14.csv'
output_folder = 'Cluster_separation_2'
clustering_input=numpy.loadtxt('input/'+input_file,delimiter=';')
clusters = numpy.asarray(sorted(clustering_input, key=lambda x: x[1]))


name_clusters= (set((clusters[:,0])))






with open('input/nc_filtered_data.csv', 'rb') as datafile:
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
    Similarity_analysis.var_similarity(output_folder,len(name_clusters),len(header_input),header_input)