"""
Creating plots according to data from the PCA
Author : Diviyan Kalainathan
Date : 1/07/2016

The goal is to plot the repartition of weights of vars to the axis according to the category
which the vars belong to
"""

import csv, numpy
import matplotlib.pyplot as plt

plot = 4  # Var for the type of plot
# 1: Graph of evolution of weight according to variables sorted by category*
# 2: graph of the most significant vars
# 3: Same as #2 but not with abs values
# 4: Correlation between Vars categories and axis.
path = 'output/aa~/raw_indent_vectors_aa_~10.csv'

points_toplot = 50
analysed_dimension = 0
plot_y = []
colors = ['0.75', 'b', 'r', 'c', 'y', 'm', 'k', 'g']
flags = True  # Separate the flags from the categories or not

# Creating a dictionary of var names and their category
path2 = 'input/filtered_data.csv'
dic = [[] for l in range(2)]
with open(path2, 'rb') as datafile:
    datareader = csv.reader(datafile, delimiter=';')
    header = next(datareader)
    for name in header:
        dic[0] += [name]
        if name[-4:] == 'flag':
            dic[1] += [0]
        else:
            dic[1] += [-1]

with open('input/Variables_info.csv', 'rb') as datafile:
    datareader = csv.reader(datafile, delimiter=',')
    header = next(datareader)

    for row in datareader:
        for i in [i for i, x in enumerate(dic[0]) if (x.rsplit('_', 1)[0] == row[0])]:
            if (flags and dic[1][i] != 0) or not flags:
                dic[1][i] = int(row[5])

        for i in [i for i, x in enumerate(dic[0]) if (dic[1][i] == -1 and (x == row[0][:-1] or x == row[0]))]:
            dic[1][i] = int(row[5])

data = numpy.loadtxt(path, delimiter=';')
print(data[15:1000, 2])
if plot == 1:
    # Sorting the dictionary
    dic[1], dic[0], idx_sort = (list(t) for t in zip(*sorted(zip(dic[1], dic[0], range(len(dic[0]))))))
    len(dic[0])
    print(dic)
    print(idx_sort)
    plot_x = [[] for l in range(len(colors))]
    separ = []
    for k in range(len(idx_sort) - 1):
        if dic[1][k] != dic[1][k + 1]:
            separ += [k + 1]
    j = 0
    for i in idx_sort:
        for l in range(len(colors)):
            plot_x[l] += [abs(data[i, l])]
        j += 1

    # Plotting the results
    for s in separ:
        plt.plot((s, s), (0, numpy.amax(plot_x)), 'k-')

    separ = [0] + separ + [len(idx_sort)]

    for co in colors:
        for s in range(len(separ) - 1):
            plot_x[colors.index(co)][separ[s]:separ[s + 1]] = sorted(plot_x[colors.index(co)][separ[s]:separ[s + 1]])
            if s == 0:
                print(plot_x[colors.index(co)][separ[s]:separ[s + 1]])

        plt.plot(range(j), plot_x[colors.index(co)], color=co)

elif plot == 2:

    for i in range(len(dic[1])):
        plot_y += [(data[i, analysed_dimension])]

    pltbar = plt.bar(range(points_toplot), plot_y[:points_toplot], align='center')
    for i in range(points_toplot):
        if plot_y[i]<0:
            dic[0][i]='-'+ dic[0][i]
    plt.xticks(xrange(points_toplot), dic[0][:points_toplot], rotation='60')
    for j in range(points_toplot):
        pltbar[j].set_color(colors[dic[1][j]])

elif plot == 3:

    abs_val = []

    for i in range(len(dic[1])):
        plot_y += [(data[i, analysed_dimension])]
        abs_val += [abs(data[i, analysed_dimension])]

    abs_val, plot_y, dic[1], dic[0] = (list(t) for t in
                                       zip(*sorted(zip(abs_val, plot_y, dic[1], dic[0]), reverse=True)))
    for i in range(points_toplot):
        if plot_y[i] < 0:
            dic[0][i] = '-' + dic[0][i]
    pltbar = plt.bar(range(points_toplot), plot_y[:points_toplot], align='center')
    plt.xticks(xrange(points_toplot), dic[0][:points_toplot], rotation='60')
    for j in range(points_toplot):
        pltbar[j].set_color(colors[dic[1][j]])

elif plot == 4:

    # Sorting the dictionary
    dic[1], idx_sort, dic[0] = (list(t) for t in zip(*sorted(zip(dic[1], range(len(dic[0])), dic[0]))))
    print(dic)
    print(idx_sort)
    separ = []

    sorted_data = [[] for l in range(len(data[0, :]))]
    for i in idx_sort:
        for l in range(len(data[0, :])):
            sorted_data[l] += [abs(data[i, l])]
            if l == 2:
                print (data[i, l])

    print(numpy.size(sorted_data))
    print(2462 * len(data[0, :]))

    for k in range(len(idx_sort) - 1):
        if dic[1][k] != dic[1][k + 1]:
            separ += [k + 1]

    separ = [0] + separ + [len(idx_sort)]

    # print numpy.sum(numpy.power(sorted_data[2][separ[1]:separ[1 + 1]], 2))

    plot_x = numpy.zeros((len(separ) - 1, len(data[0, :])))  # len(separ)-1 is the number of groups in vars

    if abs(numpy.sum(numpy.power(sorted_data[0], 2)) - 1) > 0.001:  # Comparing floats is tricky
        vp = True
    else:
        vp = False

    for l in range(len(data[0, :])):
        # print(numpy.size(sorted_data[l]))
        for i in range(len(separ) - 1):
            # if i==1:
            # print(numpy.sum(numpy.power(sorted_data[l][separ[i]:separ[i+1]],2)))
            plot_x[i, l] = numpy.sum(numpy.power(sorted_data[l][separ[i]:separ[i + 1]], 2))/(separ[i+1]-separ[i])
        if vp:
            plot_x[:, l] /= numpy.sum(numpy.power(sorted_data[l], 2))
            # print(numpy.sum(numpy.power(sorted_data[0], 2)))
    if not flags:
        plot_x[0,:]=0
    print(vp)
    print(separ)
    print dic[0][separ[1]:separ[2]]
    print idx_sort[separ[1]:separ[2]]

    xlegend = ['Flags', 'Professional activity', 'Work time organization'
        , 'Constraints, prevention & accidents', 'Work organization'
        , 'Health', 'Background & career', 'Self survey']
    plt.matshow(plot_x)
    plt.title('Correlation matrix of weights between categories and dimensions')
    plt.xlabel('Dimensions')
    plt.ylabel('Categories')
    plt.yticks(xrange(len(plot_x[:, 0])), xlegend, rotation='0')

    plt.colorbar()

plt.show()
