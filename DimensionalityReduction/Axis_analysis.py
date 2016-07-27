import v_test
import csv

with open('output/prepared_data.csv', 'rb') as datafile:
    var_reader = csv.reader(datafile, delimiter=';')
    header = next(var_reader)


v_test.v_test('output/prep_numpyarray.csv', 'output/ws+/computed_data_w_svp+_5.csv', header,'ws_vp')
