"""
Code made by David Lopez-Paz
All credits to Mr. Lopez-Paz
"""

from common import *

if(int(sys.argv[1])==0):
  w   = np.random.randn(3000,4)
  f,i = featurize('data/pairs.csv',w)
  l   = np.genfromtxt('data/targets.csv')
  pickle.dump((f,l,w), open("pickles/train_features_3000.pkl", "wb"))

else:
  x_tr,y_tr,w = pickle.load(open("pickles/train_features_3000.pkl","rb"))

  grid = {
    'n_estimators'     : [500,1000,5000],
    'max_features'     : [None],
    'min_samples_leaf' : [3,4,5,10,20], # controls complexity of weak learners
    'max_depth'        : [50]
  }

  clf1 = GridSearchCV(GBC(), grid, n_jobs=32, verbose=3, scoring='roc_auc')
  clf1.fit(x_tr,y_tr==+1);
  
  clf2 = GridSearchCV(GBC(), grid, n_jobs=32, verbose=3, scoring='roc_auc')
  clf2.fit(x_tr,y_tr==-1);

  print(clf1.best_params_)
  print(clf1.best_score_)
  print(clf2.best_params_)
  print(clf2.best_score_)

  pickle.dump((clf1,clf2,w), open("pickles/classifier.pkl", "wb"))
