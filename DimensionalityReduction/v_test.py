'''
Analyses the axis, by clustering and returns v-type of vars
Author : Diviyan Kalainathan
Date : 28/06/2016

'''
import csv, numpy, heapq, operator, os


def v_test(input_data, computed_data, list_vars,folder):
    """
    :param input_data: Data used to do the PCA(String)
    :param computed_data: Data after applying PCA(String)
    :param list_vars:List of these vars(list[String])
    :param folder: Folder of output files(String)
    :return: 0
    """
    numpy.set_printoptions(threshold='nan')
    print('-Value-Type axis analysis-')
    print('--Import Data')
    pca_data = numpy.loadtxt(computed_data, delimiter=';')
    prep_data = numpy.loadtxt(input_data, delimiter=';')
    num_total_items = pca_data.shape[1]
    num_sample_items =3000.0
    print(num_total_items)

    print 'Shape PCA data : ', pca_data.shape
    print 'Shape Prep data: ', prep_data.shape
    high_values = [[] for i in range(pca_data.shape[0])]
    low_values = [[] for i in range(pca_data.shape[0])]
    print 'Done.'
    print('--Computing total mean and std values')
    total_mean = numpy.zeros((prep_data.shape[0]))
    total_std = numpy.zeros((prep_data.shape[0]))
    for ques in range(prep_data.shape[0]):
        total_mean[ques] = numpy.mean(prep_data[ques, :])
        total_std[ques] = numpy.std(prep_data[ques, :])

    print 'Done.'
    print '--Sorting highest values and computing v-test values'
    for dim in range(pca_data.shape[0]):
        print'--- Dim : ', dim
        high_values[dim] = list(zip(*heapq.nlargest(num_sample_items, enumerate(pca_data[dim, :]), key=operator.itemgetter(1)))[0])
        low_values[dim] = list(zip(*heapq.nsmallest(num_sample_items, enumerate(pca_data[dim, :]), key=operator.itemgetter(1)))[0])
        output_high = numpy.zeros((prep_data.shape[0], num_sample_items))
        output_low = numpy.zeros((prep_data.shape[0], num_sample_items))

        for idx in range(len(high_values[dim])):
            output_high[:, idx] = prep_data[:, high_values[dim][idx]]
            output_low[:, idx] = prep_data[:, low_values[dim][idx]]

        if not os.path.exists('output/axis_analysis/'+folder+'/axis_' + str(dim)):
            os.makedirs('output/axis_analysis/'+folder+'/axis_' + str(dim))
        numpy.savetxt('output/axis_analysis/'+folder+'/axis_' + str(dim) + '/high_values.csv', output_high, delimiter=';')
        numpy.savetxt('output/axis_analysis/'+folder+'/axis_' + str(dim) + '/low_values.csv', output_low, delimiter=';')

        with open('output/axis_analysis/'+folder+'/axis_' + str(dim) + '/v-type.csv', 'wb') as outputfile:
            datawriter = csv.writer(outputfile, delimiter=';', quotechar='|')
            datawriter.writerow(['Var name', 'V-type High', 'V-Type Low'])

        for n_var in range(len(list_vars)):
            result = [list_vars[n_var],
                      ((numpy.mean(output_high[n_var, :]) - total_mean[n_var]) / numpy.sqrt(((num_total_items-num_sample_items)/(num_total_items-1))*((total_std[n_var]**2)/num_sample_items))),
                      ((numpy.mean(output_low[n_var, :]) - total_mean[n_var]) / numpy.sqrt(((num_total_items-num_sample_items)/(num_total_items-1))*((total_std[n_var]**2)/num_sample_items)))]  # ! Calcul v-type

            with open('output/axis_analysis/'+folder+'/axis_' + str(dim) + '/v-type.csv', 'a') as outputfile:
                datawriter = csv.writer(outputfile, delimiter=';', quotechar='|',
                                        lineterminator='\n')
                datawriter.writerow(result)
    print 'Done !'
    return 0
