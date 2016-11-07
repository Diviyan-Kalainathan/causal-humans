"""
From causality-treated data; construct a graph of causality
Author : Diviyan Kalainathan
Date : 7/11/2016
"""

import csv
import cPickle as pkl
import numpy
import sys
import skeleton_construction_methods as scm


inputfolder = 'output/obj8/pca_var/cluster_5/'
input_publicinfo = inputfolder+'publicinfo_c_5.csv'
causal_results = inputfolder + 'results_lp_CSP+Public_thres0.12.csv'  # csv with 3 cols, Avar, Bvar & target
pairsfile= inputfolder + 'pairs_c_5.csv'

