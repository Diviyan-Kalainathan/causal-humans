


import numpy,csv,re


n_features=2462
inputdata = numpy.zeros((n_features, 31112))

with open('output/prepared_data.csv', 'rb') as datafile:
    var_reader = csv.reader(datafile, delimiter=';')
    header = next(var_reader)
    num_row = 0
    for row in var_reader:
        for i in range(0, n_features):
            if (not re.search('[a-zA-Z]', row[i])) :

                    inputdata[i, num_row] = float(row[i])
            else:
                inputdata[i, num_row] = 0

        num_row += 1
        if num_row % 5000 == 0:
            print('.')
        elif num_row % 50 == 0:
            print('.'),

numpy.savetxt('output/prep_numpyarray.csv',inputdata, delimiter=';')