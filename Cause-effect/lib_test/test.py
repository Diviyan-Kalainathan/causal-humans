from fastkde import fastKDE
import pylab as PP
from numpy import *
from sklearn.preprocessing import scale
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from scipy.stats import pearsonr
import multiprocessing as mp




def evalerrorpredict(x,y):


    numPoints = 128+1
    # numPoints = 33

    pX, axes = fastKDE.pdf(x, numPoints=numPoints)
    pOfYGivenX, axes = fastKDE.conditional(y, x, numPoints=numPoints)


    totalError = 0

    for i in range(0, len(axes[0])):
        minError = 9999999

        for jprim in range(0, len(axes[1])):
            error = 0
            for j in range(0, len(axes[1])):
                if(pOfYGivenX[j,i] > 0):
                    error += (axes[1][j] - axes[1][jprim])**2 * pOfYGivenX[j,i]

            minError = min(minError, error)

        totalError += minError*pX[i]


    return totalError


def task_pred(inputfilespath,outputfilespath):
    numtest = 2
    f = open(inputfilespath);
    pairs = f.readlines();
    pairs.pop(0)
    f.close();

    p_te=[]

    for k in range(0, len(pairs)):

        if(k%100 == 0):
            print(k)

        r = pairs[k].split(",", 2)

        x = scale(np.array(r[1].split(), dtype=np.float))
        y = scale(np.array(r[2].split(), dtype=np.float))


        errorpredictYgivenX = evalerrorpredict(x,y)
        errorpredictXgivenY = evalerrorpredict(y,x)

        p_te += [errorpredictXgivenY /errorpredictYgivenX - 1]


    df = pd.read_csv(inputfilespath, index_col="SampleID")

    Results = pd.DataFrame(index=df.index)

    Results['Target'] = p_te
    Results.to_csv(outputfilespath + str(numtest) + ".csv", sep=';', encoding='utf-8')


    sys.stdout.write('Generated output file ' + outputfilespath)
    sys.stdout.flush()




def predict(data, results, max_proc):

    print("start predict ")


    pool = mp.Pool(processes=max_proc)

    for idx in range(len(data)):
        print('Data ' + str(idx))
        pool.apply_async(task_pred, args=(data[idx], results[idx]))
    pool.close()
    pool.join()



# fig,axs = PP.subplots(1,2,figsize=(10,5))
#
# #Plot a scatter plot of the incoming data
# axs[0].plot(x,y,'k.',alpha=0.1)
# axs[0].set_title('Original (x,y) data')
#
# #Set axis labels
# for i in (0,1):
#     axs[i].set_xlabel('x')
#     axs[i].set_ylabel('y')



# #Draw a contour plot of the conditional
# axs[1].contourf(axes[0],axes[1],pOfYGivenX,64)
# #Overplot the original underlying relationship
#
# axs[1].set_title('P(y|x)')
#
# #Set axis limits to be the same
# xlim = [amin(axes[0]),amax(axes[0])]
# ylim = [amin(axes[1]),amax(axes[1])]
# axs[1].set_xlim(xlim)
# axs[1].set_ylim(ylim)
# axs[0].set_xlim(xlim)
# axs[0].set_ylim(ylim)
#
# fig.tight_layout()
#
# PP.savefig('conditional_demo.png')
# PP.show()