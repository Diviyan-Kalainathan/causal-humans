import v_test
import csv

with open('output/prepared_data.csv', 'rb') as datafile:
    var_reader = csv.reader(datafile, delimiter=';')
    header = next(var_reader)

# Load var info
num_bool = []
spec_note = []
type_var = []
color_type = []
category = []
obj_subj=[]

with open('input/Variables_info.csv', 'rb') as datafile:
    var_reader = csv.reader(datafile, delimiter=',')
    header_var = next(var_reader)
    for var_row in var_reader:
        type_var += [var_row[1]]
        num_bool += [var_row[3]]
        spec_note += [var_row[4]]
        category += [int(var_row[5])]
        obj_subj += [var_row[6]]

    category_type = []
    obj_subj_type = []
    row_len = 0
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


v_test.v_test('output/prep_numpyarray.csv', 'output/std+1/ws+/computed_data_w_svp+_5.csv', header,'ws_vp',color_type)
