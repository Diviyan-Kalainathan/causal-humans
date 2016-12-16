# -*- coding: utf-8 -*-
'''
Plotting results / V-test analysis
Author : Diviyan Kalainathan
Date : 4/08/2016
'''

import csv, numpy, heapq, os
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from itertools import cycle
import complex_radar as CR


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


obj=False

syn_axis_a = []
syn_axis_a.append({  ## ??"contr_ry":[], # Intensité du travail
    "poly": [1, 1, 0],
    "objectif": [1, 0],
    "objmodif": [1, 1, 1, 0],
    "objattein": [1, 1, 0, 0],
    "procedur": [1, 0],
    "debord": [1, 0],
    "contrad": [1, 0],
    ## ?? "attention":[],
    "a1i": [0, 0, 1, 1, 0],
    "depech": [1, 1, 0, 0],
    "a2a": [1, 1, 0, 0],
    "a2b": [1, 1, 0, 0],
    "a2c": [1, 1, 0, 0],
    "restmai": [1, 0]})

syn_axis_a.append({"detresse": [1, 0],  # exigences émotionnelles
                   "calmer": [1, 0],
                   "tension4": [1, 0],
                   "public": [1, 0],
                   "a2i": [1, 1, 0, 0],
                   "a2j": [1, 1, 0, 0],
                   "a2k": [1, 1, 0, 0],
                   "b4a": [1, 0],
                   "b4b": [1, 0]})

syn_axis_a.append({"comment": [0, 1],  # autonomie
                   "stark": [0, 1, 1],
                   "incident": [1, 0, 0],
                   "delais": [1, 0],
                   "interup": [1, 0],
                   "a1j": [0, 0, 1, 1],
                   "monoton": [0, 0, 1, 1],
                   "b5e": [0, 0, 1, 1],
                   "a2g": [1, 1, 0, 0],
                   "nouvelle": [1, 0],
                   "a1l": [0, 0, 1, 1],
                   "computil1": [1, 0],
                   "computil2": [1, 0],
                   "chaine": [0, 1],
                   "repete": [0, 1],
                   "cycle": [0, 1]})

syn_axis_a.append({"aidchef": [1, 0],  # rapports sociaux
                   "aidcoll": [1, 0],
                   "aidautr": [1, 0],
                   "aidext": [1, 0],
                   "aidom": [1, 0],
                   "travseul": [0, 0, 1, 1],
                   "chgtcoll": [1, 0, 0],
                   "collect": [1, 0],
                   "acchef": [0, 0, 1, 1],
                   "accol": [0, 0, 1, 1],
                   "tension2": [0, 1],
                   "tension3": [0, 1],
                   "tension4": [0, 1],
                   "encadr": [1, 1, 0],
                   "a1a": [0, 0, 1, 1],
                   "a1b": [0, 0, 1, 1],
                   "a1e": [0, 0, 1, 1],
                   "a1f": [0, 0, 1, 1],
                   "b1a": [0, 1],
                   "b1b": [0, 1],
                   "b1c": [0, 1],
                   "b1d": [0, 1],
                   "b1e": [0, 1],
                   "b1f": [0, 1],
                   "b1g": [0, 1],
                   "b1h": [0, 1],
                   "b1i": [0, 1],
                   "b1j": [0, 1],
                   "b4c": [0, 1],
                   "b4d": [0, 1],
                   "b5f": [1, 1, 0, 0]})

syn_axis_a.append({"a2d": [1, 1, 0, 0],  # Conflits de valeur
                   "a2h": [1, 1, 0, 0],
                   "b5a": [0, 0, 1, 1],
                   "b5b": [0, 0, 1, 1],
                   "b5c": [1, 1, 0, 0],
                   "corrtan": [0, 1],
                   "corrinf": [0, 1],
                   "corrcop": [0, 1],
                   "corrcol": [0, 1],
                   "corrlog": [0, 1],
                   "corrmat": [0, 1],
                   "corrform": [0, 1]})

syn_axis_a.append({"crainte": [1, 0],  # Insécurité économique et ses changements
                   "metier": [1, 0],
                   "nochom": [0, 1],
                   "tenir": [0, 1],
                   "souhait": [0, 1],
                   "fortmod1": [1, 0],
                   "fortmod2": [1, 0],
                   "fortmod3": [1, 0],
                   "fortmod4": [1, 0],
                   "fortmod5": [1, 0],
                   "fortmod6": [1, 0],
                   "fortmod7": [1, 0],
                   "changop": [0, 1],
                   "chgtinfo": [0, 1],
                   "chgtcons": [0, 1],
                   "chgtinfl": [0, 1],
                   "a2e": [0, 0, 1, 1],
                   "a2f": [1, 1, 0, 0, ],
                   "b5d": [1, 1, 0, 0]})

syn_axis_a.append({"a1g": [0, 0, 1, 1],  # Reconaissance, rémunération et evaluation
                   "a1h": [0, 0, 1, 1],
                   "payecom": [1, 1, 1, 0, 0],
                   "sieg34": [1, 0],
                   "a1c": [0, 0, 1, 1],
                   "a1d": [0, 0, 1, 1],
                   "a2l": [1, 1, 0, 0],
                   "eva": [1, 0],
                   "evacrit": [1, 0]})

syn_axis_a.append({"cwdebou": [1, 0],  # Contraintes physiques
                   "cwpostu": [1, 0],
                   "cwlourd": [1, 0],
                   "cwdepla": [1, 0],
                   "cwmvt": [1, 0],
                   "cwvib": [1, 0],
                   ## ? "contr_envt":[],
                   "entendr": [1, 0, 0],
                   "secfupou": [1, 0],
                   "sectoxno": [1, 0],
                   "secinfec": [1, 0],
                   "secaccid": [1, 0],
                   "secrout": [1, 0],
                   "conduite": [1, 0]})

syn_axis_a.append({  ## ? "h_hebdo":[], # Contraintes horaires organisation du temps de travail
                    "repos": [0, 1],
                    "samedi": [1, 1, 0],
                    "dimanche": [1, 1, 0],
                    "horangt": [0, 1],
                    "previs": [1, 1, 0, 0],
                    "horvar": [0, 1, 1, 1],
                    "periode": [1, 0],
                    "ptmatin": [1, 1, 0],
                    "soir": [1, 1, 0],
                    "nuit": [1, 1, 0],
                    "hsup": [1, 1, 0, 0],
                    "astreinte": [1, 0],
                    # "jourtr":[], ##>5 ??
                    "controle": [0, 1, 1, 1, 1],
                    "joindre": [1, 0]})

obj_names = ['Indep.', u'Santé', 'Ouvriers', u'CSP+Privé', 'ServPart', 'CSP+Public', 'Immigr.', 'Accid.']
subj_names = ['RAS', 'Tens.Col', 'Indep.', 'Heur.', 'Tens.Hie', 'Chgts']
permutation_obj = [0, 4, 6, 2, 3, 5, 1, 7]
permutation_subj = [2, 3, 0, 5, 1, 4]

# Computing values of synth. axis on clusters
pd.options.mode.chained_assignment = None  # default='warn'
permutation = True


def compute_axis_value(df_o, axis_def):
    axis_vars = []
    for key in axis_def:
        axis_vars.append(key)
    # Refilter df
    df_column_names = [name for name in df_o.columns if any(name.startswith(key + '_') for key in axis_vars)]
    df = df_o[df_column_names]
    print(df_column_names)

    # Apply operations on columns
    for var in axis_def:
        for subq in range(len(axis_def[var])):
            df[var + '_' + str(subq + 1)] = df[var + '_' + str(subq + 1)].apply(lambda x: x * axis_def[var][subq])

        # Combining
        df[var] = sum([df[var + '_' + str(subq + 1)] for subq in range(len(axis_def[var]))])

        # Managing missing values: replacing with mean
        df.loc[df[var + '_flag'] == 0, var] = df.loc[df[var + '_flag'] != 0, var].mean()
        df.pop(var + '_flag')  # Removing var_flag

        for subq in range(len(axis_def[var])):
            df.pop(var + '_' + str(subq + 1))  # Removing sub_variables of questions

    # Finally, sum all variables
    df['axis_s_i'] = sum([df[var] for var in axis_def])

    return df['axis_s_i']


df_chunk = pd.read_csv('input/m_prepared_data.csv', sep=';', chunksize=10 ** 4, low_memory=False)
df_data = pd.DataFrame()
for chunk in df_chunk:
    df_data = pd.concat([df_data, chunk])
df_data.reset_index()

var_names = []
for axis in syn_axis_a:
    for key in axis:
        var_names.append(key)
var_names = set(var_names)  # Take unique names

# Filter DF
df_column_names = [name for name in df_data.columns if any(
    name.startswith(key + '_') for key in var_names)]  ## Select all variables that we are going to use
df_data = df_data[df_column_names]
print(df_column_names)

synth_axis = pd.DataFrame()
for axis in range(len(syn_axis_a)):
    synth_axis['axis_s' + str(axis)] = compute_axis_value(df_data, syn_axis_a[axis])

result_folder = 'output/synthetic_axis/'

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# Save temp result
synth_axis.to_csv(result_folder + "synthetic_axis_tmp.csv")

# Mean values over clusters

if permutation:
    obj_names = [obj_names[elt] for elt in permutation_obj]
    subj_names = [subj_names[elt] for elt in permutation_subj]

axis_values_sorted = [[[[] for i in range(len(subj_names))] for j in range(len(obj_names))] for k in
                      range(len(syn_axis_a))]

clustering_o = 'cluster_predictions_c8_n500_r12-obj.csv'
clustering_s = 'cluster_predictions_c6_n500_r12-subj.csv'

clustering_input_o = numpy.loadtxt('input/' + clustering_o, delimiter=';')
clusters_o = numpy.asarray(sorted(clustering_input_o, key=lambda x: x[1]))

clustering_input_s = numpy.loadtxt('input/' + clustering_s, delimiter=';')
clusters_s = numpy.asarray(sorted(clustering_input_s, key=lambda x: x[1]))

for idx, row in synth_axis.iterrows():
    for axis_number in range(len(row)):
        axis_values_sorted[axis_number][int(clusters_o[idx, 0])][int(clusters_s[idx, 0])].append(
            row[axis_number])  # reclassing values in order according to : axis, cluster o & s

result_matrixes = [numpy.zeros((len(obj_names), len(subj_names))) for k in range(len(syn_axis_a))]

for k in range(len(syn_axis_a)):
    for j in range(len(obj_names)):
        for i in range(len(subj_names)):
            result_matrixes[k][j, i] = numpy.mean(axis_values_sorted[k][j][i])

# Permute elts according to chosen order:
if permutation:
    for matrix in result_matrixes:
        matrix[range(len(obj_names))] = matrix[permutation_obj]
        matrix[:, range(len(subj_names))] = matrix[:, permutation_subj]

for k in range(len(syn_axis_a)):
    numpy.savetxt(result_folder + 'synth_axis_np' + str(k) + '.csv', result_matrixes[k], delimiter=';')

## Print results
# ex : objective clusters
if obj:
    names=obj_names
else:
    names=subj_names

data_o = [[] for i in range(len(names))]
for matrix in result_matrixes:
    for i in range(len(data_o)):
        if obj:
            data_o[i].append(numpy.mean(matrix[i]))
        else: data_o[i].append(numpy.mean(matrix[:,i]))

# Normalize
# (numpy.min([data_o[j][i] for j in range(len(data_o))]),
ranges = [
    [(numpy.max([data_o[j][i] for j in range(len(data_o))])), numpy.std([data_o[j][i] for j in range(len(data_o))])] for
    i in range(len(result_matrixes))]

for i in range(len(data_o)):
    for axis in range(len(result_matrixes)):
        data_o[i][axis] = data_o[i][axis] / ranges[axis][1]
ranges = [
    [(numpy.max([data_o[j][i] for j in range(len(data_o))])), numpy.std([data_o[j][i] for j in range(len(data_o))])] for
    i in range(len(result_matrixes))]

for i in range(len(data_o)):
    for axis in range(len(result_matrixes)):
        data_o[i][axis] = data_o[i][axis] / ranges[axis][0]

f=plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
fig_leg=plt.figure()

axis = (u'                  Intensité \n         du travail',
        u'          Exigences Emotionnelles',
        u'Autonomie',
        u'Rapports Sociaux',
        u'Conflits de valeur                 ',
        u'Insécurité économique                      \n et ses changements             ',
        u'Reconnaissance      ,\n remunération et évaluation     ',
        'Contraintes Physiques',
        '                  Contraintes horaires \n                      et organisation du temps de travail')

# ranges=((0.1,1),(0.1,1.1),(0.1,1.1),(0.1,1.1),(0.1,1.1),(0.1,1.1),(0.1,1.1),(0.1,1.1),(0.1,1.1))
ranges = [
    (numpy.min([data_o[j][i] for j in range(len(data_o))])-0.1, (numpy.max([data_o[j][i] for j in range(len(data_o))])+0.05)) for
    i in range(len(result_matrixes))]

print(ranges)
radar = CR.ComplexRadar(f, axis, ranges)
handles = []
colors = cycle(
    ['cyan', 'indigo', 'seagreen', 'gold', 'blue', 'darkorange', 'red', 'grey', 'darkviolet', 'mediumslateblue'])
for curve, color,idx in zip(data_o, colors,range(len(data_o))):
    radar.plot(curve,color=color)
    handles.append(matplotlib.lines.Line2D([], [], color=color, label=names[idx]))

fig_leg.legend(handles=handles,labels=names,loc='best')
plt.show()
