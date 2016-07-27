#DEPRECATED



import csv
import numpy
# Load var info


researched_var='le'
researched_index=0

result=numpy.zeros((31112,1))

count=0
with open('output/prepared_data.csv', 'rb') as datafile:
    datareader = csv.reader(datafile, delimiter=';', quotechar='"')
    num_row = 0

    header=next(datareader)

    for i in [i for i, x in enumerate(header) if x == researched_var]:
        researched_index=i
        print('Found! ')

    print(researched_index)
    for row in datareader:
        if row[researched_index]!='0':
            print(row[researched_index])
    '''for row in datareader:
        if int(row[researched_index])==1:
            result[count]=1
        elif  int(row[researched_index+1])==1:
            result[count]=2
        elif int(row[researched_index + 2]) == 1:
            result[count] = 3
        elif int(row[researched_index + 3]) == 1:
            result[count] = 4
        elif int(row[researched_index + 4]) == 1:
            result[count] = 5
        elif int(row[researched_index + 5]) == 1:
            result[count]= 6
        elif int(row[researched_index + 6]) == 1:
            result[count]= 7
        elif int(row[researched_index + 7]) == 1:
            result[count]= 8
        elif int(row[researched_index + 8]) == 1:
            result[count]= 9
        elif int(row[researched_index + 9]) == 1:
            result[count]= 10
        elif int(row[researched_index + 10]) == 1:
            result[count]= 11
        elif int(row[researched_index + 11]) == 1:
            result[count]= 12
        elif int(row[researched_index + 12]) == 1:
            result[count]= 13
        elif int(row[researched_index + 13]) == 1:
            result[count]= 14
        elif int(row[researched_index + 14]) == 1:
            result[count]= 15
        elif int(row[researched_index + 15]) == 1:
            result[count]= 16
        elif int(row[researched_index + 16]) == 1:
            result[count]= 17
        elif int(row[researched_index + 17]) == 1:
            result[count]= 18
        count+=1
    
numpy.savetxt('output/counter_csei.csv',result,delimiter=';')'''
