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



f = open("data/train_pairs.csv");
pairs = f.readlines();
pairs.pop(0)
f.close();

y_te = np.genfromtxt("data/train_target.csv", delimiter=",")

# for k in range(0, len(y_te)):

k = 2000

maxstd = 2


print(k+1)
print("True causal " + str(y_te[k]))

r = pairs[k].split(",", 2)

x = scale(np.array(r[1].split(), dtype=np.float))
y = scale(np.array(r[2].split(), dtype=np.float))


mask = (x > -maxstd) & (x < maxstd) & ( y > -maxstd) & ( y < maxstd)
x = x[mask]
y = y[mask]


numPoints = 32

intx = np.round(x*(numPoints)/maxstd/2)
inty = np.round(y*(numPoints)/maxstd/2)

pXY = np.zeros((numPoints,numPoints))

for i in range(len(intx)):
    coordx = min( intx[i], numPoints/2 - 1) + numPoints/2
    coordy = min( inty[i], numPoints/2 - 1) + numPoints/2

    pXY[coordx,coordy] += 1

maxPXY = amax(pXY)

for i in range(numPoints):
    for j in range(numPoints):
        pXY[i,j] = pXY[i,j]/maxPXY


fig,axs = PP.subplots(1,2,figsize=(10,5))

#Plot a scatter plot of the incoming data
axs[0].plot(x,y,'k.',alpha=0.1)
axs[0].set_title('Original (x,y) data')



axeX = np.arange(0,numPoints)
axeY = np.arange(0,numPoints)

#Draw a contour plot of the conditional
axs[1].matshow(pXY)
#Overplot the original underlying relationship


#Set axis limits to be the same
# xlim = [amin(axes[0]),amax(axes[0])]
# ylim = [amin(axes[1]),amax(axes[1])]
# axs[1].set_xlim(xlim)
# axs[1].set_ylim(ylim)
# axs[0].set_xlim(xlim)
# axs[0].set_ylim(ylim)

fig.tight_layout()

# PP.savefig('conditional_demo.png')
PP.show()