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

print(len(x))

mask = (x > -maxstd) & (x < maxstd) & ( y > -maxstd) & ( y < maxstd)
x = x[mask]
y = y[mask]



numPoints = 32+1

pXY, axes = fastKDE.pdf(x, y, numPoints=numPoints,axisExpansionFactor = 0.1)


fig,axs = PP.subplots(1,2,figsize=(10,5))

#Plot a scatter plot of the incoming data
axs[0].plot(x,y,'k.',alpha=0.1)
axs[0].set_title('Original (x,y) data')

#Set axis labels
for i in (0,1):
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('y')



# arrayImage = np.ravel(pXY)

#Draw a contour plot of the conditional
axs[1].contourf(axes[0],axes[1],pXY,64)
#Overplot the original underlying relationship

axs[1].set_title('P(x,y)')

#Set axis limits to be the same
xlim = [amin(axes[0]),amax(axes[0])]
ylim = [amin(axes[1]),amax(axes[1])]
axs[1].set_xlim(xlim)
axs[1].set_ylim(ylim)
axs[0].set_xlim(xlim)
axs[0].set_ylim(ylim)

fig.tight_layout()

# PP.savefig('conditional_demo.png')
PP.show()