'''
Generate plots from var info
24/05/2016

'''

import csv, numpy
import matplotlib.pyplot as plt

num_bool = []
spec_note = []
type_var = []
color_type = []
category = []
obj_subj=[]
category_type = []
obj_subj_type =[]
mode=2
flags=False
if mode==1:
    with open('input/Variables_info.csv', 'rb') as datafile:
        var_reader = csv.reader(datafile, delimiter=',')
        header_var = next(var_reader)
        for var_row in var_reader:
            type_var += [var_row[1]]
            num_bool += [var_row[3]]
            spec_note += [var_row[4]]
            category += [int(var_row[5])]
            obj_subj += [var_row[6]]

    row_len=0

    percent_obj=numpy.zeros((8))
    percent_subj=numpy.zeros((8))

    for num_col in range(0, 541):
        if spec_note[num_col] != 'I':
            if type_var[num_col] == 'C' or (type_var[num_col] == 'D' and spec_note[num_col] == 'T'):
                row_len += 2  #
                color_type += ['C']
                color_type += ['FC']
                category_type += [category[num_col], category[num_col]]
                obj_subj_type += [obj_subj[num_col], [obj_subj[num_col]]]


            elif type_var[num_col] == 'D' and spec_note[num_col] != '-2' and spec_note[num_col] != 'T':
                # print(num_col)
                row_len += int(num_bool[num_col]) + 1
                for i in range(0, int(num_bool[num_col])):
                    color_type += ['D']
                    category_type += [category[num_col]]
                    obj_subj_type += [obj_subj[num_col]]

                color_type += ['FD']
                category_type += [category[num_col]]
                obj_subj_type += [obj_subj[num_col]]

    total = len(category)
    print 'Objectives :' , obj_subj.count('O')
    print 'Subjectives :' , obj_subj.count('S')
    '''
    total = len(category_type)

    for i in range(8):

        sum_obj=0
        sum_subj=0
        for j  in [j for j, x in enumerate(category_type) if x == i]:
            if obj_subj_type[j]=='O':
                sum_obj+=1
            elif obj_subj_type[j]=='S':
                sum_subj+=1
        percent_obj[i]= (float(sum_obj)/total)*100
        percent_subj[i]=(float(sum_subj)/total)*100

    if not flags:
        percent_obj=percent_obj[1:]
        percent_subj=percent_subj[1:]
    '''
    for i in range(8):

        sum_obj=0
        sum_subj=0
        for j  in [j for j, x in enumerate(category) if x == i]:
            if obj_subj[j]=='O':
                sum_obj+=1
            elif obj_subj[j]=='S':
                sum_subj+=1
        percent_obj[i]= (float(sum_obj)/total)*100
        percent_subj[i]=(float(sum_subj)/total)*100

    if not flags:
        percent_obj=percent_obj[1:]
        percent_subj=percent_subj[1:]

    N=7
    ind = numpy.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, percent_obj, width, color='b')


    rects2 = ax.bar(ind + width, percent_subj, width, color='r')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Proportion des types de questions (%)')
    ax.set_title('Proportion des types de questions en fonction des categories')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(['Activite\n professionnelle/ \n statut', 'Organisation du \ntemps de travail'
            , 'Contraintes \nphysiques, \nprevention et accidents', 'Organisation du travail'
            , 'Sante', 'Parcours familial \net professionnel', 'Risques \n pyschosociaux'])#'Drapeaux',

    ax.legend((rects1[0], rects2[0]), ('Objectives', 'Subjectives'))

elif mode==2:
    with open('input/datacsv.csv','rb') as inputfile:
        reader=csv.reader(inputfile,delimiter=';')
        header=next(reader)
        print(len(header))
        dataind=[]
        count=0
        for row in reader:
            count+=1
            for data in row:
                if data==''or data=='NA':
                    dataind+=[0]
                else:
                    dataind+=[1]
    print(numpy.mean(dataind))



def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

plt.show()
