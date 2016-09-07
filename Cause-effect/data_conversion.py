"""
Convert data into the desired form
Author : Diviyan Kalainathan
Date : 28/06/2016
"""
import csv, numpy, re, os

# Input : filtered data only : no conversion yet

inputfile = "input/nc_filtered_data.csv"
inputcluster = 'cluster_predictions_c6_n500_r12-subj.csv'
inputclusterpath = 'input/' + inputcluster
converteddatapath = 'input/c-e_converted_data.csv'
conversion = False
outputfolder = 'output/subj6/'

# init of lists
name_var = []
type_var = []
# description=[]
num_bool = []
spec_note = []

# Import variable data
print('--Loading data & parameters--')
with open('input/Variables_info.csv', 'rb') as datafile:
    var_reader = csv.reader(datafile, delimiter=',', quotechar='\'')
    header = next(var_reader)

    for var_row in var_reader:
        name_var += [var_row[0]]
        type_var += [var_row[1]]
        num_bool += [var_row[3]]
        spec_note += [var_row[4]]

# initialize the sparse matrix of flags
if conversion:
    with open('input/nc_filtered_data.csv', 'rb') as datafile:
        input_length = sum(1 for row in datafile) - 1
        print('Lines to process : ' + repr(input_length))
        row_len = 0
        for num_col in range(0, 541):
            if spec_note[num_col] != 'I' and spec_note[num_col] != "-1":
                if type_var[num_col] == 'C' or (type_var[num_col] == 'D' and spec_note[num_col] == 'T'):
                    row_len += 2  #
                elif type_var[num_col] == 'D' and spec_note[num_col] != '-2' and spec_note[num_col] != 'T':
                    # print(num_col)
                    row_len += 2

                    # the matrix is now implemented in the csv result
                    # This computation is now made in order to cross-check the columns
        print 'Row no : ', row_len

    print('Done.')
    print('--Processing data--')
    with open('input/nc_filtered_data.csv', 'rb') as datafile:
        # Warning: the first var must not be a flag

        datareader = csv.reader(datafile, delimiter=';', quotechar='"')
        num_row = 0

        # Reading variables to convert according to type

        header = next(datareader)
        output_header = []
        count = 0
        count2 = 0
        for row in datareader:

            if count > 10:
                print('.'),
                count2 += 1
                count = 0
            else:
                count += 1
            if count2 > 100:
                print('.')
                count2 = 0

            result_row = []
            # print(datareader.line_num),

            flag_vector = numpy.zeros((541, 1))
            for num_col in range(0, 541):

                if spec_note[num_col] != 'I':  # if is not an ignored value

                    if type_var[num_col] == 'C' and spec_note[num_col] == 'T':  # T : Time var

                        valid_value = False
                        if row[num_col + 1] != '' and row[num_col + 1] != 'NA':  # If there is valid data
                            min_val = 0
                            valid_value = True

                            if name_var[num_col] == 'infoh':  # convert to min per day

                                temp_result = int(row[num_col + 1])
                                for i in [i for i, x in enumerate(name_var) if x == 'infohu1']:
                                    if row[i + 1] != '' and row[i + 1] != 'NA':
                                        if int(row[i + 1]) == 2:
                                            temp_result *= 60
                                    else:
                                        valid_value = False

                                for i in [i for i, x in enumerate(name_var) if x == 'infohu2']:
                                    if row[i + 1] != '' and row[i + 1] != 'NA':

                                        if int(row[i + 1]) == 2:
                                            temp_result /= 7
                                        elif int(row[i + 1]) == 3:
                                            temp_result /= 30
                                    else:
                                        valid_value = False

                                if valid_value:
                                    result_row += [str(temp_result)]
                            else:
                                for i in [i for i, x in enumerate(name_var) if x == (name_var[num_col])[:-1] + 'm']:

                                    if row[i + 1] != '' and row[i + 1] != 'NA':
                                        min_val = i

                                        if name_var[num_col] == 'debdeta':

                                            result_row += [str(12 * int(row[num_col + 1]) + int(row[min_val + 1]))]
                                            # the ID row has to be left out

                                        else:
                                            result_row += [str(60 * int(row[num_col + 1]) + int(
                                                row[min_val + 1]))]  # the ID row has to be left out
                                    else:
                                        valid_value = False

                        if valid_value:  # if the value is valid, the flag is put to 1
                            if flag_vector[num_col] >= 0:
                                flag_vector[num_col] = 1
                                result_row += [str(1)]


                            else:
                                result_row += [str(0)]

                        else:
                            flag_vector[num_col] = -1
                            result_row += [str(0)]  # data
                            result_row += [str(0)]  # no answer var

                            # flag_vector[num_row, len(result_row) - 1] = -1

                        if num_row == 0:
                            output_header += [(name_var[num_col])[:-1]]
                            output_header += [(name_var[num_col])[:-1] + '_flag']

                    elif type_var[num_col] == 'C' and spec_note[
                        num_col] != '-1':  # The var is non-discrete and not ignored
                        # Copy the value
                        if row[num_col + 1] != '' and row[num_col + 1] != 'NA':
                            if name_var[num_col] == 'coeffuc' or name_var[num_col] == 'jourtr':
                                if flag_vector[num_col] >= 0:
                                    flag_vector[num_col] = 1
                                    result_row += [str(int(float(row[num_col + 1]) * 10))]  # from floats to integers
                                    result_row += [str(1)]  # Flag value
                                else:
                                    result_row += [str(0)]
                                    result_row += [str(0)]

                            else:

                                if flag_vector[num_col] >= 0:
                                    flag_vector[num_col] = 1
                                    result_row += [row[num_col + 1]]  # the ID row has to be left out
                                    result_row += [str(1)]  # Flag value
                                else:
                                    result_row += [str(0)]
                                    result_row += [str(0)]
                        else:
                            result_row += [str(0)]
                            result_row += [str(0)]
                            flag_vector[num_col] = -1

                        if num_row == 0:
                            output_header += [name_var[num_col]]
                            output_header += [name_var[num_col] + '_flag']

                    # ToDo : Special case for booleans : modify var_info and recode values in 0/1

                    elif type_var[num_col] == 'D' and spec_note[num_col] != '-1':  # var is discrete (D) and not ignored
                        # Warning : first var must not be a flag

                        if spec_note[num_col] == '-2':  # var is a flag
                            # Look for var ref
                            for i in [i for i, x in enumerate(name_var) if x == (name_var[num_col])[:-5]]:
                                if row[num_col + 1] != '' and row[num_col + 1] != 'NA':
                                    if row[num_col + 1] < 1:
                                        flag_vector[i] = -1
                                    else:
                                        flag_vector[i] = 1

                                    for j in [j for j, y in enumerate(output_header)
                                              if y == (name_var[num_col])[:-5] + '_flag']:
                                        if j < len(result_row):
                                            if row[num_col + 1] < 1:
                                                result_row[j] = str(0)

                                            else:
                                                result_row[j] = str(1)


                        elif spec_note[num_col] == 'T':  # If the var is time var

                            valid_value = False
                            if row[num_col + 1] != '' and row[num_col + 1] != 'NA':
                                valid_value = True
                                if name_var[num_col] == 'nbrkmu':
                                    for i in [i for i, x in enumerate(name_var) if x == (name_var[num_col])[:-1]]:
                                        if row[i + 1] != '' and row[i + 1] != 'NA':
                                            var_val = int(row[i + 1])
                                            if row[num_col + 1] == '1':
                                                result_row += [str(var_val * 52)]
                                            elif row[num_col + 1] == '2':
                                                result_row += [str(var_val * 12)]
                                            else:
                                                result_row += [str(var_val)]
                                        else:
                                            valid_value = False


                                elif name_var[num_col] == 'finetudu':
                                    for i in [i for i, x in enumerate(name_var) if x == (name_var[num_col])[:-1]]:
                                        if row[i + 1] != '' and row[i + 1] != 'NA':
                                            var_val = int(row[i + 1])
                                            if row[num_col + 1] == '2':
                                                for i in [i for i, x in enumerate(name_var) if x == 'anais']:
                                                    result_row += [str(var_val - int(row[i + 1]))]
                                            else:
                                                result_row += [str(var_val)]
                                        else:
                                            valid_value = False


                                elif name_var[num_col] == 'congeu':
                                    for i in [i for i, x in enumerate(name_var) if x == (name_var[num_col])[:-1]]:
                                        if row[i + 1] != '' and row[i + 1] != 'NA':

                                            var_val = int(row[i + 1])
                                            if row[num_col + 1] == '2':
                                                result_row += [str(var_val * 7)]
                                            else:
                                                result_row += [str(var_val)]
                                        else:
                                            valid_value = False


                                elif name_var[num_col] == 'tpsintu':
                                    for i in [i for i, x in enumerate(name_var) if x == (name_var[num_col])[:-1]]:
                                        if row[i + 1] != '' and row[i + 1] != 'NA':

                                            var_val = int(row[i + 1])
                                            if row[num_col + 1] == '1':
                                                result_row += [str(var_val * 365)]
                                            elif row[num_col + 1] == '2':
                                                result_row += [str(var_val * 30)]
                                            elif row[num_col + 1] == '3':
                                                result_row += [str(var_val * 7)]
                                            else:
                                                result_row += [str(var_val)]
                                        else:
                                            valid_value = False

                                elif name_var[num_col] == 'dudetu':
                                    for i in [i for i, x in enumerate(name_var) if x == (name_var[num_col])[:-1]]:
                                        if row[i + 1] != '' and row[i + 1] != 'NA':

                                            var_val = int(row[i + 1])
                                            if row[num_col + 1] == '1':
                                                result_row += [str(var_val * 365)]
                                            elif row[num_col + 1] == '2':
                                                result_row += [str(var_val * 30)]
                                            elif row[num_col + 1] == '3':
                                                result_row += [str(var_val * 7)]
                                            else:
                                                result_row += [str(var_val)]

                                        else:
                                            valid_value = False


                                elif name_var[num_col] == 'rabspu':
                                    for i in [i for i, x in enumerate(name_var) if x == (name_var[num_col])[:-1]]:
                                        if row[i + 1] != '' and row[i + 1] != 'NA':
                                            var_val = int(row[i + 1])
                                            if row[num_col + 1] == '1':
                                                result_row += [str(var_val * 365)]
                                            elif row[num_col + 1] == '2':
                                                result_row += [str(var_val * 30)]
                                            elif row[num_col + 1] == '3':
                                                result_row += [str(var_val * 7)]
                                            else:
                                                result_row += [str(var_val)]
                                        else:
                                            valid_value = False

                            if valid_value == True:
                                if flag_vector[num_col] >= 0:
                                    flag_vector[num_col] = 1
                                    result_row += [str(1)]


                            else:
                                result_row += [str(0)]
                                result_row += [str(0)]
                                flag_vector[num_col] = -1

                            if num_row == 0:
                                output_header += [(name_var[num_col])[:-1]]
                                output_header += [(name_var[num_col])[:-1] + '_flag']


                        elif spec_note[num_col] == 'A':
                            # Special rules, individual cases
                            valid_value = False
                            min_var_val = 1
                            if (row[num_col + 1] != '' and row[num_col + 1] != 'NA') or name_var[num_col] == 'tranchre':

                                var_val = -1
                                # if name_var[num_col] == 'ageq': ##useful?

                                if (name_var[num_col])[:-2] == 'lien_' or (name_var[num_col])[:-2] == 'lienpr':
                                    if int(row[num_col + 1]) == 0:
                                        var_val = 0
                                    elif int(row[num_col + 1]) == 01:
                                        var_val = 1
                                    elif int(row[num_col + 1]) == 02:
                                        var_val = 2
                                    elif int(row[num_col + 1]) == 03:
                                        var_val = 3
                                    elif int(row[num_col + 1]) == 10:
                                        var_val = 4
                                    elif int(row[num_col + 1]) == 21:
                                        var_val = 5
                                    elif int(row[num_col + 1]) == 22:
                                        var_val = 6
                                    elif int(row[num_col + 1]) == 31:
                                        var_val = 7
                                    elif int(row[num_col + 1]) == 32:
                                        var_val = 8
                                    elif int(row[num_col + 1]) == 40:
                                        var_val = 9
                                    elif int(row[num_col + 1]) == 50:
                                        var_val = 10
                                    elif int(row[num_col + 1]) == 60:
                                        var_val = 11
                                    elif int(row[num_col + 1]) == 90:
                                        var_val = 12

                                elif ((name_var[num_col]) == 'qual_emplx' or (name_var[num_col]) == 'qual_siret' or
                                              (name_var[num_col]) == 'qual_adr'):
                                    if row[num_col + 1] == 'A':
                                        var_val = 0
                                    elif row[num_col + 1] == 'B':
                                        var_val = 1
                                    elif (name_var[num_col]) == 'qual_adr':
                                        if row[num_col + 1] == 'C':
                                            var_val = 2
                                        elif row[num_col + 1] == 'D':
                                            var_val = 3

                                elif ((name_var[num_col])[:-3] == 'lnais' or (name_var[num_col]) == 'lnaisd' or
                                              (name_var[num_col])[:-3] == 'natnais' or (name_var[num_col]) == 'nati'):

                                    if int(row[num_col + 1]) == 11:
                                        var_val = 0
                                    elif int(row[num_col + 1]) == 21:
                                        var_val = 1
                                    elif int(row[num_col + 1]) == 22:
                                        var_val = 2
                                    elif int(row[num_col + 1]) == 23:
                                        var_val = 3
                                    elif int(row[num_col + 1]) == 24:
                                        var_val = 4
                                    elif int(row[num_col + 1]) == 31:
                                        var_val = 5
                                    elif int(row[num_col + 1]) == 34:
                                        var_val = 6
                                    elif int(row[num_col + 1]) == 35:
                                        var_val = 7
                                    elif int(row[num_col + 1]) == 41:
                                        var_val = 8
                                    elif int(row[num_col + 1]) == 42:
                                        var_val = 9
                                    elif int(row[num_col + 1]) == 43:
                                        var_val = 10
                                    elif int(row[num_col + 1]) == 44:
                                        var_val = 11
                                    elif int(row[num_col + 1]) == 45:
                                        var_val = 12
                                    elif int(row[num_col + 1]) == 46:
                                        var_val = 13
                                    elif int(row[num_col + 1]) == 51:
                                        var_val = 14

                                elif (name_var[num_col]) == 'doublontype':
                                    if (row[num_col + 1]) == 'PCP_EXT':
                                        var_val = 0
                                    elif (row[num_col + 1]) == 'EXT EXT':
                                        var_val = 1


                                elif (name_var)[num_col] == 'csei':
                                    if row[num_col + 1] != ' ' and not re.search('[a-zA-Z]', row[num_col + 1]):
                                        if int(row[num_col + 1]) == 10:
                                            var_val = 0
                                        elif int(row[num_col + 1]) == 21:
                                            var_val = 1
                                        elif int(row[num_col + 1]) == 22:
                                            var_val = 2
                                        elif int(row[num_col + 1]) == 23:
                                            var_val = 3
                                        elif int(row[num_col + 1]) == 31:
                                            var_val = 4
                                        elif int(row[num_col + 1]) == 32:
                                            var_val = 5
                                        elif int(row[num_col + 1]) == 36:
                                            var_val = 6
                                        elif int(row[num_col + 1]) == 41:
                                            var_val = 7
                                        elif int(row[num_col + 1]) == 46:
                                            var_val = 8
                                        elif int(row[num_col + 1]) == 47:
                                            var_val = 9
                                        elif int(row[num_col + 1]) == 48:
                                            var_val = 10
                                        elif int(row[num_col + 1]) == 51:
                                            var_val = 11
                                        elif int(row[num_col + 1]) == 54:
                                            var_val = 12
                                        elif int(row[num_col + 1]) == 55:
                                            var_val = 13
                                        elif int(row[num_col + 1]) == 56:
                                            var_val = 14
                                        elif int(row[num_col + 1]) == 61:
                                            var_val = 15
                                        elif int(row[num_col + 1]) == 66:
                                            var_val = 16
                                        elif int(row[num_col + 1]) == 69:
                                            var_val = 17


                                elif (name_var[num_col])[0:4] == 'cser':
                                    if row[num_col + 1] != ' ' and row[num_col + 1] != '0' and not re.search('[a-zA-Z]',
                                                                                                             row[
                                                                                                                         num_col + 1]):
                                        if int(row[num_col + 1]) < 10 and (row[num_col + 1]) > 0:
                                            var_val = int(row[num_col + 1])

                                elif (name_var[num_col])[0:3] == 'fap':
                                    var_val = -1
                                elif (name_var[num_col]) == 'naf4':
                                    if (row[num_col + 1]) == 'ES':
                                        var_val = 0
                                    elif (row[num_col + 1]) == 'ET':
                                        var_val = 1
                                    elif (row[num_col + 1]) == 'EU':
                                        var_val = 2
                                    elif (row[num_col + 1]) == 'EV':
                                        var_val = 3

                                elif (name_var[num_col]) == 'naf17':

                                    if (row[num_col + 1]) == 'AZ':
                                        var_val = 0
                                    elif (row[num_col + 1]) == 'C1':
                                        var_val = 1
                                    elif (row[num_col + 1]) == 'C2':
                                        var_val = 2
                                    elif (row[num_col + 1]) == 'C3':
                                        var_val = 3
                                    elif (row[num_col + 1]) == 'C4':
                                        var_val = 4
                                    elif (row[num_col + 1]) == 'C5':
                                        var_val = 5
                                    elif (row[num_col + 1]) == 'DE':
                                        var_val = 6
                                    elif (row[num_col + 1]) == 'FZ':
                                        var_val = 7
                                    elif (row[num_col + 1]) == 'GZ':
                                        var_val = 8
                                    elif (row[num_col + 1]) == 'HZ':
                                        var_val = 9
                                    elif (row[num_col + 1]) == 'IZ':
                                        var_val = 10
                                    elif (row[num_col + 1]) == 'JZ':
                                        var_val = 11
                                    elif (row[num_col + 1]) == 'KZ':
                                        var_val = 12
                                    elif (row[num_col + 1]) == 'LZ':
                                        var_val = 13
                                    elif (row[num_col + 1]) == 'MN':
                                        var_val = 14
                                    elif (row[num_col + 1]) == 'OQ':
                                        var_val = 15
                                    elif (row[num_col + 1]) == 'RU':
                                        var_val = 16

                                elif (name_var[num_col])[0:4] == 'prof' or (name_var[num_col]) == 'pe':
                                    var_val = -1
                                elif (name_var[num_col])[0:3] == 'cse':
                                    if row[num_col + 1] != ' ' and not re.search('[a-zA-Z]', row[num_col + 1]):
                                        if int(row[num_col + 1]) == 11:
                                            var_val = 0
                                        elif int(row[num_col + 1]) == 12:
                                            var_val = 1
                                        elif int(row[num_col + 1]) == 13:
                                            var_val = 2
                                        elif int(row[num_col + 1]) == 21:
                                            var_val = 3
                                        elif int(row[num_col + 1]) == 22:
                                            var_val = 4
                                        elif int(row[num_col + 1]) == 23:
                                            var_val = 5
                                        elif int(row[num_col + 1]) == 31:
                                            var_val = 6
                                        elif int(row[num_col + 1]) == 33:
                                            var_val = 7
                                        elif int(row[num_col + 1]) == 34:
                                            var_val = 8
                                        elif int(row[num_col + 1]) == 35:
                                            var_val = 9
                                        elif int(row[num_col + 1]) == 37:
                                            var_val = 10
                                        elif int(row[num_col + 1]) == 38:
                                            var_val = 11
                                        elif int(row[num_col + 1]) == 42:
                                            var_val = 12
                                        elif int(row[num_col + 1]) == 43:
                                            var_val = 13
                                        elif int(row[num_col + 1]) == 44:
                                            var_val = 14
                                        elif int(row[num_col + 1]) == 45:
                                            var_val = 15
                                        elif int(row[num_col + 1]) == 46:
                                            var_val = 16
                                        elif int(row[num_col + 1]) == 47:
                                            var_val = 17
                                        elif int(row[num_col + 1]) == 48:
                                            var_val = 18
                                        elif int(row[num_col + 1]) == 52:
                                            var_val = 19
                                        elif int(row[num_col + 1]) == 53:
                                            var_val = 20
                                        elif int(row[num_col + 1]) == 54:
                                            var_val = 21
                                        elif int(row[num_col + 1]) == 55:
                                            var_val = 22
                                        elif int(row[num_col + 1]) == 56:
                                            var_val = 23
                                        elif int(row[num_col + 1]) == 62:
                                            var_val = 24
                                        elif int(row[num_col + 1]) == 63:
                                            var_val = 25
                                        elif int(row[num_col + 1]) == 64:
                                            var_val = 26
                                        elif int(row[num_col + 1]) == 65:
                                            var_val = 27
                                        elif int(row[num_col + 1]) == 67:
                                            var_val = 28
                                        elif int(row[num_col + 1]) == 68:
                                            var_val = 29
                                        elif int(row[num_col + 1]) == 69:
                                            var_val = 30

                                elif (name_var[num_col]) == 'peun':
                                    var_val = -1

                                elif (name_var[num_col]) == 'peun10':
                                    if not re.search('[a-zA-Z]', row[num_col + 1]):
                                        if int(row[num_col + 1]) < 10 and (row[num_col + 1]) >= 0:
                                            var_val = int(row[num_col + 1])

                                elif (name_var[num_col]) == 'activfin':
                                    list_miss_values = [4, 34, 40, 44, 48, 54, 57, 67, 76, 83, 89]
                                    activfin_val = int(row[num_col + 1])
                                    if activfin_val == 0:
                                        var_val = -1
                                    else:
                                        inc = 0
                                        for missed_val in list_miss_values:
                                            if activfin_val > missed_val:
                                                inc += 1

                                        var_val = activfin_val - inc

                                elif (name_var[num_col]) == 'tranchre':
                                    for idx in [idx for idx, x in enumerate(name_var) if x == 'revmen']:
                                        if row[idx + 1] != '' and row[idx + 1] != 'NA':
                                            revmen = int(row[idx + 1])
                                            list_tranchre = [400, 600, 800, 1000, 1200, 1500, 1800, 2000, 2500, 3000,
                                                             4000,
                                                             6000, 10000]
                                            var_val = 1
                                            for tranch in list_tranchre:
                                                if revmen > tranch:
                                                    var_val += 1

                                if var_val >= 0:
                                    valid_value = True
                                    result_row += [str(var_val)]
                                else:
                                    result_row += [str(0)]

                                if valid_value:

                                    result_row += [str(1)]
                                    flag_vector[num_col] = 1

                                else:
                                    result_row += [str(0)]
                                    flag_vector[num_col] = -1

                            else:
                                result_row += [str(0)]  # Value
                                result_row += [str(0)]  # Flag
                                flag_vector[num_col] = -1

                            if num_row == 0:
                                output_header += [name_var[num_col]]
                                output_header += [name_var[num_col] + '_flag']


                        else:
                            # Basic rules
                            valid_value = False
                            if spec_note[num_col] == '0':
                                min_var_val = 0
                            else:
                                min_var_val = 1
                            if row[num_col + 1] != '' and row[num_col + 1] != 'NA':

                                if int(row[num_col + 1]) in range(min_var_val, min_var_val + int(num_bool[num_col])):
                                    result_row += [str(int(row[num_col + 1]))]
                                    valid_value = True
                                else:
                                    result_row += [str(0)]
                            else:
                                result_row += [str(0)]

                            if valid_value:
                                result_row += [str(1)]
                                flag_vector[num_col] = 1

                            else:
                                result_row += [str(0)]
                                flag_vector[num_col] = -1
                            if num_row == 0:
                                output_header += [name_var[num_col]]
                                output_header += [name_var[num_col] + '_flag']

            if num_row == 0:
                with open(converteddatapath, 'wb') as sortedfile:
                    datawriter = csv.writer(sortedfile, delimiter=';', quotechar='|')
                    datawriter.writerow(output_header)

            with open(converteddatapath, 'a') as sortedfile:
                datawriter = csv.writer(sortedfile, delimiter=';', quotechar='|',
                                        lineterminator='\n')
                datawriter.writerow(result_row)

            num_row += 1

    # flag_vector (-1) values conversion
    print('')
    print('Calculated Row_length : ' + repr(row_len))
    print('Output header length : ' + repr(len(output_header)))
    print 'Writed lines : ', num_row
    print('Done.')

'''print('--Sparse matrix conversion--')

flag_vector[flag_vector == -1] = 0

for i in range(0, 10):
    for j in range(0, row_len):
        print(flag_vector[i, j]),

    print('')

S = flag_vector.todense()
print('Done.')
print('--Saving data--')
numpy.savetxt("output/flag_vector.csv", S, fmt='%i', delimiter=',')
print('Done.')'''  # No sparse matrix conversion

print 'Separating into multiple files according to clustering : ', inputcluster

cluster_index = numpy.loadtxt(inputclusterpath, delimiter=';')
cluster_index = numpy.asarray(sorted(cluster_index, key=lambda x: x[1]))
clusters = list(set((cluster_index[:, 0])))
clusters = [int(clu) for clu in clusters]
with open(converteddatapath, 'rb') as inputfile:
    datareader = csv.reader(inputfile, delimiter=';', quotechar='|')
    c_header = next(datareader)

    # Create files/folders
    for num_c in range(len(clusters)):
        if not os.path.exists(outputfolder + 'cluster_' + str(clusters[num_c])):
            os.makedirs(outputfolder + 'cluster_' + str(clusters[num_c]))

        with open(outputfolder + 'cluster_' + str(clusters[num_c]) + '/data_c_' + str(clusters[num_c]) + '.csv',
                  'wb') as sortedfile:
            datawriter = csv.writer(sortedfile, delimiter=';', quotechar='|')
            datawriter.writerow(c_header)

    # Separate into files
    line = 0
    for row in datareader:
        with open(outputfolder + 'cluster_' + str(int(cluster_index[line, 0])) + '/data_c_' + str(
                int(cluster_index[line, 0])) + '.csv', 'a') as sortedfile:
            datawriter = csv.writer(sortedfile, delimiter=';', quotechar='|',
                                    lineterminator='\n')
            datawriter.writerow(row)

        print '-- line : ', line
        line += 1

print ('End of program')
