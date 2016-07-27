'''
Analyses the clusters and returns the similar values in vars
Author : Diviyan Kalainathan
Date : 28/06/2016
'''

import csv

def var_similarity(data_folder,num_clusters, num_vars, list_vars):
    """
    :param data_folder: Folder where the clustering output is(String)
    :param num_clusters: Number of clusters(int)
    :param num_vars:Number of variables to analyse(int)
    :param list_vars:List of these vars(list[String])
    :return: 0
    """

    for num_file in range(num_clusters):
        with open('output/' + data_folder + '/cluster_similarity_' + str(int(num_file)) + '.csv', 'wb') as outputfile:
            datawriter = csv.writer(outputfile, delimiter=';', quotechar='|')
            datawriter.writerow(['Var name','Most_occ value','Percent'])

        for n_var in range(num_vars):
            with open('output/'+ data_folder +'/cluster_'+str(num_file)+'.csv', 'rb') as datafile:
                datareader = csv.reader(datafile, delimiter=';', quotechar='|')
                header = next(datareader)
                name_value=[]
                count=[]
                if n_var==0:
                    num_line=0
                for row in datareader:
                    if n_var==0:
                        num_line+=1
                    if row[n_var] not in name_value:
                        name_value+=[row[n_var]]
                        count+=[1]
                    else:
                        for i in [ i for i,x in enumerate(name_value) if x == row[n_var]]:
                            count[i]+=1
                #print(float(max(count))/num_line)
                result=[list_vars[n_var],name_value[count.index(max(count))], str(float(max(count))/num_line)]
                with open('output/' + data_folder + '/cluster_similarity_' + str(int(num_file)) + '.csv', 'a') as outputfile:
                    datawriter = csv.writer(outputfile, delimiter=';', quotechar='|',
                                            lineterminator='\n')
                    datawriter.writerow(result)

    return 0