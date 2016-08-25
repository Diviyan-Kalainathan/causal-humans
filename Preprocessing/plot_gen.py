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


with open('input/Variables_info2.csv', 'rb') as datafile:
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


for i in range(8):
    total = category_type.count(i)
    sum_obj=0
    for j  in [j for j, x in enumerate(category_type) if x == i]:
        if obj_subj_type[j]=='O':
            sum_obj+=1

    percent_obj[i]= (float(sum_obj)/total)*100

percent_subj=100-percent_obj

N=8
ind = numpy.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, percent_obj, width, color='b')


rects2 = ax.bar(ind + width, percent_subj, width, color='r')

# add some text for labels, title and axes ticks
ax.set_ylabel('Proportion des types de questions')
ax.set_title('Proportion des types de questions en fonction des categories')
ax.set_xticks(ind + width)
ax.set_xticklabels(['Drapeaux', 'Activite\n professionnelle', 'Organisation du \ntemps de travail'
        , 'Contraintes \nphysiques, \nprevention et accidents', 'Organisation du travail'
        , 'Sante', 'Parcours familial \net professionnel', 'Risques \n pyschosociaux'])

ax.legend((rects1[0], rects2[0]), ('Objectives', 'Subjectives'))


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()
