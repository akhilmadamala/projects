from os import listdir
from numpy import array
from numpy import vstack
from numpy.linalg import lstsq
from pandas import read_csv
from matplotlib import pyplot

def regress(y):
    X = array([i for i in range(len(y))]).reshape(len(y), 1)

    b = lstsq(X, y)[0][0]

    yhat = b * X[:, 0]
    return yhat



sequences, targets, groups, paths = load_dataset()



paths = [1, 2, 3, 4, 5, 6]
seq_paths = dict()
for path in paths:
    seq_paths[path] = [sequences[j] for j in range(len(paths)) if paths[j] == path]

pyplot.figure()
for i in paths:
    pyplot.subplot(len(paths), 1, i)
    # line plot each variable
    for j in [0, 1, 2, 3]:
        pyplot.plot(seq_paths[i][0][:, j], label='Anchor ' + str(j + 1))
    pyplot.title('Path ' + str(i), y=0, loc='left')
pyplot.show()

seq = sequences[0]
variables = [0, 1, 2, 3]
pyplot.figure()
for i in variables:
    pyplot.subplot(len(variables), 1, i + 1)
    # plot the series
    pyplot.plot(seq[:, i])
    # plot the trend
    pyplot.plot(regress(seq[:, i]))
pyplot.show()