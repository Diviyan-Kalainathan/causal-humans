'''
Preprocessing data for principal component analysis on converted & filtered data
Author : Diviyan Kalainathan
Date : 1/06/2016

The mean value is put for non-discrete data for which the flag is 0
'''

import csv,numpy,re

# init of lists
name_var = []
type_var = []
# description=[]
num_bool = []
spec_note = []


with open('input/Variables_info.csv', 'rb') as datafile:
    var_reader = csv.reader(datafile, delimiter=',')
    header_var = next(var_reader)
    for var_row in var_reader:
        if var_row[1]=='C':
            name_var += [var_row[0]]
            type_var += [var_row[1]]
            num_bool += [var_row[3]]
            spec_note += [var_row[4]]


with open('input/filtered_data.csv', 'rb') as datafile:
    datareader = csv.reader(datafile, delimiter=';')
    header_input = next(datareader)

    flags_index = []
    c_var_index = []

    #Finding the indexes
    for c_var in name_var:
        for i in [i for i, x in enumerate(header_input) if x == c_var]:
            c_var_index+=[i]
        for i in [i for i, x in enumerate(header_input) if x == c_var + '_flag']:
            flags_index+=[i]

    print(len(header_input))
    inputdata= numpy.zeros((31112,len(header_input)))
    c_var_mean=numpy.zeros((1,len(c_var_index)))
    #Compute the mean values
    num_row=0
    for row in datareader:

        for i in range(0, 2462):

            if not re.search('[a-zA-Z]', row[i]): ##Must recode values : Re is not to be used !
                if '.' not in row[i]:
                    inputdata[num_row,i] = int(row[i])
                else:
                    inputdata[num_row,i] = float(row[i])
            else:
                inputdata[num_row,i] = 0
        num_row += 1
        if num_row % 5000 == 0:
            print('.')
        elif num_row % 50 == 0:
            print('.'),

    indx=0
    for j in c_var_index:
        print(j),
        print(flags_index[indx])

        c_var_mean[0,indx]=numpy.average(inputdata[:,j],weights=(inputdata[:,flags_index[indx]]).astype(bool))
        indx+=1

with open('output/prepared_data.csv', 'wb') as outputfile:
    datawriter = csv.writer(outputfile, delimiter=';', quotechar='|')
    datawriter.writerow(header_input)



    #Replace values
with open('input/filtered_data.csv', 'rb') as datafile:
    datareader = csv.reader(datafile, delimiter=';')
    header_input = next(datareader)
    for row in datareader:
        indx=0
        for flag in flags_index:
            if int(row[flag])==0:
                row[c_var_index[indx]]=str(c_var_mean[0,indx])
            indx+=1

        with open('output/prepared_data.csv', 'a') as outputfile:
            datawriter = csv.writer(outputfile, delimiter=';', quotechar='|',
                                    lineterminator='\n')
            datawriter.writerow(row)


print(c_var_index)
print(flags_index)


