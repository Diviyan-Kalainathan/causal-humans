"""
Code made by David Lopez-Paz
All credits to Mr. Lopez-Paz
"""

import sys, pickle, numpy as np
from   sklearn.preprocessing    import scale
from   sklearn.ensemble         import GradientBoostingClassifier as GBC
from   sklearn.cross_validation import train_test_split
from   sklearn.metrics          import auc_score
from   sklearn.grid_search      import GridSearchCV

def score(y,p):
  return (auc_score(y==1,p)+auc_score(y==-1,-p))/2

def featurize_row(row,w):
  x   = scale(np.fromstring(row.split(",",2)[1],dtype=np.float,sep=" "))
  y   = scale(np.fromstring(row.split(",",2)[2],dtype=np.float,sep=" "))
  xy  = scale(x*y)
  return np.maximum(np.dot(w,np.vstack((x,y,xy,np.ones(x.shape)))),0).mean(1)

def featurize(filename,w):
  f = open(filename);
  pairs = f.readlines();
  f.close();
  del pairs[0];
  f   = np.array([featurize_row(row,w) for row in pairs])
  idx = [row.split(",")[0] for row in pairs]
  return (f,idx);
