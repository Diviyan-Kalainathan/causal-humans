import math
import numpy


def sigmoid(x):
  if numpy.abs(x)<705:
    return (1 / (1 + math.exp(-x)))
  elif x>=705:
    return 1
  else:
    return 0
  #return 1/(1-x)

sigmoid = numpy.vectorize(sigmoid)

inputdata = numpy.loadtxt('input/computed_data5dim_2.csv',delimiter=';')
numpy.set_printoptions(threshold='nan')

print(numpy.amin(inputdata))
print(numpy.amax(inputdata))
print(inputdata)
print(numpy.shape(inputdata))
inputdata=inputdata.astype(numpy.float)
outputdata = numpy.transpose(sigmoid(inputdata))

numpy.savetxt('input/sigmoid_data5dim.csv',outputdata,delimiter=';')