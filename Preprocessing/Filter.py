'''
Extract data from .csv file, and sort
Author : Diviyan Kalainathan
24/05/2016

'''

import csv
#Extraction of data

with open('output/converted_data.csv', 'rb') as datafile:
    datareader = csv.reader(datafile,delimiter=';')
    header = next(datareader)
    writedlines=0
    print(len(header))

#Defining the filters
    with open('output/n_filtered_data.csv', 'wb') as sortedfile:
        datawriter = csv.writer(sortedfile, delimiter=';', quotechar='|')
        datawriter.writerow(header)
    print(header)
    for i in [i for i, x in enumerate(header) if x == 'situa']:
        print(i)
        situa1_index=i

    '''for i in [i for i, x in enumerate(header) if x == 'situa_flag']:
        print(i)
        situa_flag = i'''

    for i in [i for i, x in enumerate(header) if x == 'repqaa']:
        print(i)
        repqaa = i

    for i in [i for i, x in enumerate(header) if x == 'lang4']:
        print(i)
        lang4 = i

#Applying the filter
    for row in datareader:
        usefuldata=True


        if row[situa1_index]!='1'  or (row[repqaa]!='1'and row[lang4]!='1'):
            usefuldata=False

        if usefuldata:
            writedlines+=1
            with open('output/n_filtered_data.csv', 'a') as sortedfile:
                datawriter = csv.writer(sortedfile, delimiter=';', quotechar='|',
                                         lineterminator='\n')
                datawriter.writerow(row)
               # print((row))


    print('readlines : ' + repr(datareader.line_num))
    print('writedlines : '+repr(writedlines))