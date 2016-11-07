# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:50:12 2016

@author: Philippe
"""

import pandas as pd
import numpy as np
import cPickle as pkl



def transformData(x):
    tansformedData = ""
    for values in x:
        tansformedData += " " + str(float(values))

    return tansformedData



path = "Dream5ChallengeData/"
filenameData = "net4_expression_data_Scerevisiae.tsv"


outputPath = "output/Dream5DataPreprocess/"
outputFilenamePairsData = "Scerevisiae.csv"


df_input = pd.read_csv(path + filenameData, sep='\t', encoding="latin-1")
df_ouput = pd.DataFrame(columns=["SampleID", "Data"])



for xName in df_input.columns.values:

    print(xName)

    x = df_input[xName].values



    xValuesParse = transformData(x)

    newLignOutput = pd.DataFrame([[xName, xValuesParse]],
                                     columns=["SampleID", "Data"])

    df_ouput = pd.concat([df_ouput, newLignOutput])


df_ouput.to_csv(outputPath + outputFilenamePairsData, index=False, encoding='utf-8', sep= ",")



