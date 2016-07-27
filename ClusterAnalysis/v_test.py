'''
Analyses the clusters and returns v-type of vars
Author : Diviyan Kalainathan
Date : 28/06/2016

'''
import csv,numpy

def v_type(input_data,data_folder,num_clusters, num_vars, list_vars):
    """
    :param input_data: Data used to do the clustering(String)
    :param data_folder: Folder where the clustering output is(String)
    :param num_clusters: Number of clusters(int)
    :param num_vars:Number of variables to analyse(int)
    :param list_vars:List of these vars(list[String])
    :return: 0
    """

    totaldata = numpy.zeros((num_vars, 2))  #0 for mean , #1 for
    for n in range(num_vars):
        col_data=[]
        with open('input/' + input_data, 'rb') as totalfile:
            datareader = csv.reader(totalfile, delimiter=';', quotechar='|')
            header = next(datareader)
            for row in datareader:
                col_data+=[row[n]]
            totaldata[n,0]=numpy.mean(col_data)
            totaldata[n,1]=numpy.std(col_data)

    for num_file in range(num_clusters):
        with open('output/' + data_folder + '/cluster_similarity_' + str(int(num_file)) + '.csv', 'wb') as outputfile:
            datawriter = csv.writer(outputfile, delimiter=';', quotechar='|')
            datawriter.writerow(['Var name','V-type'])


        for n_var in range(num_vars):
            with open('output/'+ data_folder +'/cluster_'+str(num_file)+'.csv', 'rb') as datafile:
                datareader = csv.reader(datafile, delimiter=';', quotechar='|')
                header = next(datareader)
                name_value=[]

                for row in datareader:
                    name_value+=[row[n_var]]

                result=[list_vars[n_var],(abs(numpy.mean(name_value)-totaldata[n_var,0])/totaldata[n_var,1])] # ! Calcul v-type
                with open('output/' + data_folder + '/cluster_similarity_' + str(int(num_file)) + '.csv', 'a') as outputfile:
                    datawriter = csv.writer(outputfile, delimiter=';', quotechar='|',
                                            lineterminator='\n')
                    datawriter.writerow(result)

    return 0