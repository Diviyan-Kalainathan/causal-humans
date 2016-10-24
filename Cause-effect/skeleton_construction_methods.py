import csv, numpy
import scipy.stats as stats
from lib.fonollosa import features
from sklearn import metrics
import pandas as pd

BINARY = "Binary"
CATEGORICAL = "Categorical"
NUMERICAL = "Numerical"
threshold_pval = 0.05

# threshold_pearsonc=0.5 #No threshold on correlation coefficient

def skel_const(mode, pairsfile, link_mat, ordered_var_names, input_publicinfo='', types=False, causal_results='',
               flags=False, epsilon_diag=0.01):
    if mode < 6:
        with open(pairsfile, 'rb') as pairs_file:
            datareader = csv.reader(pairs_file, delimiter=';')
            header = next(datareader)

            if types:
                typesfile = open(input_publicinfo, 'rb')
                typereader = csv.reader(typesfile, delimiter=';')
                type_header = next(typereader)

            var_1 = 0
            var_2 = 0
            # Idea: go through the vars and unlink the skipped (not in the pairs file) pairs of vars.
            for row in datareader:
                try:
                    types_row = next(typereader)
                except NameError:
                    pass
                if row == []:  # Skipping blank lines
                    continue

                pair = row[0].split('-')

                if not flags and ('flag' in pair[0] or 'flag' in pair[1]):
                    continue  # Skipping values w/ flags

                # Finding the pair var_1 var_2 corresponding to the line
                # and un-linking skipped values
                while pair[0] != ordered_var_names[var_1]:
                    if var_2 != len(ordered_var_names):
                        link_mat[var_1, var_2 + 1:] = 0
                    var_1 += 1
                    var_2 = 0

                skipped_value = False  # Mustn't erase checked values
                while pair[1] != ordered_var_names[var_2]:
                    if skipped_value:
                        link_mat[var_1, var_2] = 0
                    var_2 += 1
                    skipped_value = True

                # Parsing values of table & removing artifacts
                var_1_value = [float(x) for x in row[1].split(' ') if x is not '']
                var_2_value = [float(x) for x in row[2].split(' ') if x is not '']

                if len(var_1_value) != len(var_2_value):
                    raise ValueError

                if mode < 3:
                    #### Pearson's correlation to remove links ####

                    if abs(stats.pearsonr(var_1_value, var_2_value)[1]) < threshold_pval:
                        if mode == 1:
                            link_mat[var_1, var_2] = abs(stats.pearsonr(var_1_value, var_2_value)[0])
                        elif mode == 2:
                            link_mat[var_1, var_2] = (stats.pearsonr(var_1_value, var_2_value)[0])
                    else:
                        link_mat[var_1, var_2] = 0
                else:
                    try:
                        var_1_type, var_2_type = types_row[1], types_row[2]

                    except NameError:
                        var_1_type, var_2_type, = NUMERICAL, NUMERICAL

                    values1, values2 = features.discretized_sequences(var_1_value, var_1_type, var_2_value, var_2_type)
                    if mode == 3:

                        contingency_table=confusion_mat(values1,values2)

                        if contingency_table.size > 0 and min(contingency_table.shape) > 1:
                            chi2, pval, dof, expd = stats.chi2_contingency(contingency_table)
                            if pval < threshold_pval:  # there is a link
                                link_mat[var_1, var_2] = 1
                            else:
                                link_mat[var_1, var_2] = 0

                        else:
                            link_mat[var_1, var_2] = 0

                    elif mode == 4:
                        link_mat[var_1, var_2] = metrics.adjusted_mutual_info_score(values1, values2)
                    elif mode == 5:
                        contingency_table=confusion_mat(values1,values2)
                        if contingency_table.size > 0 and min(contingency_table.shape) > 1:

                            '''df= pd.DataFrame({'val1':values1,'val2':values2})
                            print df
                            confusion_matrix = pd.crosstab(df['val1'], df['val2'])
                            print confusion_matrix'''
                            link_mat[var_1, var_2] = cramers_corrected_stat(contingency_table)
                        else:
                            link_mat[var_1, var_2] = 0
        try:
            typesfile.close()
        except NameError:
            pass

        # Symmetrize matrix
        for col in range(0, (len(ordered_var_names) - 1)):
            for line in range(col + 1, (len(ordered_var_names))):
                link_mat[line, col] = link_mat[col, line]

        # Diagonal elts
        for diag in range(0, (len(ordered_var_names))):
            link_mat[diag, diag] = epsilon_diag  # To guarantee non-singularity


    elif mode == 6:
        #### Causality score to remove links ####
        with open(causal_results, 'rb') as pairs_file:
            datareader = csv.reader(pairs_file, delimiter=';')
            header = next(datareader)
            threshold = 0.12
            var_1 = 0
            var_2 = 0
            # Idea: go through the vars and unlink the skipped (not in the pairs file) pairs of vars.
            for row in datareader:

                if not flags and ('flag' in row[0] or 'flag' in row[1]):
                    continue  # Skipping values w/ flags

                # Finding the pair var_1 var_2 corresponding to the line
                # and un-linking skipped values
                while row[0] != ordered_var_names[var_1]:
                    if var_2 != len(ordered_var_names):
                        link_mat[var_1, var_2 + 1:] = 0
                    var_1 += 1
                    var_2 = 0

                skipped_value = False  # Mustn't erase checked values
                while row[1] != ordered_var_names[var_2]:
                    if skipped_value:
                        link_mat[var_1, var_2] = 0
                    var_2 += 1
                    skipped_value = True

                if float(row[2]) > threshold:
                    link_mat[var_1, var_2] = float(row[2])

        # Anti-symmetrize matrix
        for col in range(0, (len(ordered_var_names) - 1)):
            for line in range(col + 1, (len(ordered_var_names))):
                link_mat[line, col] = -link_mat[col, line]

        # Diagonal elts
        for diag in range(0, (len(ordered_var_names))):
            link_mat[diag, diag] = 0

        else:
            raise ValueError

    return link_mat


def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """

    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return numpy.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

def confusion_mat(val1,val2):
    '''
    contingency_table = numpy.zeros((len(set(val1)), len(set(val2))))
    for i in range(len(val1)):
        contingency_table[list(set(val1)).index(val1[i]),
                          list(set(val2)).index(val2[i])] += 1'''


    contingency_table= numpy.asarray(pd.crosstab(numpy.asarray(val1,dtype='object'),numpy.asarray( val2, dtype='object')))
    # Checking and sorting out bad columns/rows
    max_len, axis_del = max(contingency_table.shape), [contingency_table.shape].index(
        max([contingency_table.shape]))
    toremove = [[], []]

    for i in range(contingency_table.shape[0]):
        for j in range(contingency_table.shape[1]):
            if contingency_table[i, j] < 4:  # Suppress the line
                toremove[0].append(i)
                toremove[1].append(j)
                continue

    for value in toremove:
        contingency_table = numpy.delete(contingency_table, value, axis=axis_del)

    return contingency_table