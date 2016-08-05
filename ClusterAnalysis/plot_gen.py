'''
Plotting results / V-test analysis
Author : Diviyan Kalainathan
Date : 4/08/2016
'''
import csv,numpy,heapq
from matplotlib import pyplot as plt

output_folder = 'Cluster_separation_3'

var_to_analyze=[('naf',17),]
inputfile='output/'+output_folder+'/numpy-v-test.csv'

mode = 3
     #1 : matrices of v-tests for some vars
     #2 : highest values of v-test per cluster
     #3 : matrices of distance between objective and subjective clusters

if mode==1:
    for var in var_to_analyze:
        with open('input/prepared_data.csv', 'rb') as datafile:
            var_reader = csv.reader(datafile, delimiter=';')
            header = next(var_reader)
            


elif mode==2:
    for var in var_to_analyze:
        with open('input/prepared_data.csv', 'rb') as datafile:
            var_reader = csv.reader(datafile, delimiter=';')
            header = next(var_reader)

    data= numpy.loadtxt(inputfile,delimiter=';')

    with open('output/' + output_folder + '/max-min_v-test.csv', 'wb') as outputfile:
        datawriter = csv.writer(outputfile, delimiter=';')
        datawriter.writerow(['V-tests'])
    max_idx=[[] for i in range(data.shape[1])]
    min_idx=[[] for i in range(data.shape[1])]

    for col in range(data.shape[1]):

        max_idx[col]= heapq.nlargest(20, range(len(data[:,col])), data[:,col].take)
        min_idx[col] = heapq.nsmallest(20, range(len(data[:, col])), data[:, col].take)

    with open('output/' + output_folder + '/max-min_v-test.csv', 'a') as outputfile:
        datawriter = csv.writer(outputfile, delimiter=';')
        datawriter.writerow(['Max values'])

        for i in range(20) :
            row=[]
            for j in range(data.shape[1]):
                idx= max_idx[j][i]
                row += [header[idx]]
            datawriter.writerow(row)

        datawriter.writerow('')
        datawriter.writerow(['Min values'])

        for i in range(20):
            row = []
            for j in range(data.shape[1]):
                idx = (min_idx[j][i])
                row+=[header[idx]]
            datawriter.writerow(row)

            #for col in range(len(data)-1):
                #n_data[row,col]=data[row][col+1]

    #for col in range(len(data[1])-1))
    print(header)
elif mode==3:
    #1 is objective
    #2 is subjective

    clustering1='cluster_predictions_c8_n500_r12-obj.csv'
    clustering2='cluster_predictions_c8_n500_r12-subj.csv'

    clustering_input_1= numpy.loadtxt('input/' + clustering1, delimiter=';')
    clusters_1 = numpy.asarray(sorted(clustering_input_1, key=lambda x: x[1]))

    clustering_input_2 = numpy.loadtxt('input/' + clustering2, delimiter=';')
    clusters_2 = numpy.asarray(sorted(clustering_input_2, key=lambda x: x[1]))

    n_clusters_1 = (set((clusters_1[:, 0])))
    n_clusters_2 = (set((clusters_2[:, 0])))

    count_matrix=numpy.zeros((len(n_clusters_1),len(n_clusters_2)))
    #1st result is intersect/union
    #2nd is intersect/min(card(Cluster_i),card(Cluster_j))
    inters_union_matrix=numpy.zeros((len(n_clusters_1),len(n_clusters_2)))
    inters_min_card_c_matrix = numpy.zeros((len(n_clusters_1),len(n_clusters_2)))

    #Counting values
    for ppl in range(len(clusters_1[:,0])):
        count_matrix[clusters_1[ppl,0],clusters_2[ppl,0]]+=1

    for row in range(len(n_clusters_1)):
        for col in range(len(n_clusters_2)):
            inters_union_matrix[row,col]=float(count_matrix[row,col])\
                                         /(sum(count_matrix[row,:])+sum(count_matrix[:,col])-count_matrix[row,col])

            inters_min_card_c_matrix[row,col]=float(count_matrix[row,col])\
                                         /min(sum(count_matrix[row,:]),sum(count_matrix[:,col]))

    plt.matshow(inters_min_card_c_matrix)
    plt.title('Matrice de croisement des clusters subjectifs sur les clusters objectifs')
    plt.xlabel('Clusters objectifs')
    plt.ylabel('Clusters subjectifs')

    plt.colorbar()
    plt.show()
