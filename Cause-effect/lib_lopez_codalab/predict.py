from common import *
import pandas as pd
import multiprocessing as mp

def task_pred(inputdata,outputdata,clf1,clf2,w):
    # print("featurize test ")
    f, i = featurize(inputdata, w)

    p_te = clf1.predict_proba(f)[:, 1] - clf2.predict_proba(f)[:, 1]

    df = pd.read_csv(inputdata, index_col="SampleID")

    Results = pd.DataFrame(index=df.index)

    Results['Target'] = p_te
    Results.to_csv(outputdata, sep=';', encoding='utf-8')

    sys.stdout.write('Generated output file '+ outputdata)
    sys.stdout.flush()

def predict(data,results,modelPath, max_proc):

    print("start predict ")

    clf1, clf2, w = pickle.load(open(modelPath + "classifier.pkl", "rb"))
    pool = mp.Pool(processes=max_proc)

    for idx in range(len(data)):
        print('Data '+str(idx))
        pool.apply_async(task_pred,args=(data[idx],results[idx],clf1,clf2,w,))
    pool.close()
    pool.join()









