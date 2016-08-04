'''
Plotting results / V-test analysis
Author : Diviyan Kalainathan
Date : 4/08/2016
'''
import csv,numpy,heapq

output_folder = 'Cluster_separation_3'

var_to_analyze=[('naf',17),]
inputfile='output/'+output_folder+'/numpy-v-test.csv'

mode = 2
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
    clustering1=''
    clustering2=''