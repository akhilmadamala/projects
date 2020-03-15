from os import listdir
from numpy import array
from numpy import vstack
from pandas import read_csv
from matplotlib import pyplot
from data_set_loader import load_dataset

sequences, targets, groups, paths = load_dataset()

class1,class2 = len(targets[targets==-1]), len(targets[targets==1])
print('Class=-1: %d %.3f%%' % (class1, class1/len(targets)*100))
print('Class=+1: %d %.3f%%' % (class2, class2/len(targets)*100))

all_rows = vstack(sequences)
pyplot.figure()
variables = [0, 1, 2, 3]
for v in variables:
	pyplot.subplot(len(variables), 1, v+1)
	pyplot.hist(all_rows[:, v], bins=20)
pyplot.show()

trace_lengths = [len(x) for x in sequences]
pyplot.hist(trace_lengths, bins=50)
pyplot.show()