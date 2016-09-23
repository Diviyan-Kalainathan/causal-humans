"""
From causality-treated data; construct a graph of causality
Author : Diviyan Kalainathan
Date : 28/06/2016
"""

import csv
import cPickle as pkl


inputdata='output/...' #csv with 3 cols, Avar, Bvar & target

list_var=[] #create blank list for name of vars
causality_links=[] #List of links between vars

#Import data, construction of data structure

with open(inputdata,'rb') as inputfile:
    reader=csv.reader(inputfile, delimiter=';')
    header=next(reader)

    for row in reader:

        if (row[0]) not in list_var:
            list_var.append(row[0])
            causality_links.append([[],[]]) #0 for parents, 1 for children

        if (row[1]) not in list_var:
            list_var.append(row[1])
            causality_links.append([[],[]]) #0 for parents, 1 for children

        if float(row[3])>0:
            causality_links[list_var.index(row[0])][1].append(list_var.index(row[1]))
            causality_links[list_var.index(row[1])][0].append(list_var.index(row[0]))

        else:
            causality_links[list_var.index(row[0])][0].append(list_var.index(row[1]))
            causality_links[list_var.index(row[1])][1].append(list_var.index(row[0]))

with open(inputdata+'causality.pkl', 'wb') as handle:
    pkl.dump(causality_links,handle)

with open(inputdata+'list_vars.pkl', 'wb') as handle:
    pkl.dump(list_var,handle)


#Creating all possible combinations:

causality_possibilites=[[[],[]] for i in causality_links]
#ToDO ASK MICHELE FOR THE DATA STRUCTURE . INFINITE POSSIBILITES CAN BE GENERATED
 #1. Generate all lists
 #2. Generate up to n parents & n children