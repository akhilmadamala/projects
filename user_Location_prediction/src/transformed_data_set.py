from os import listdir
from numpy import array
from numpy import savetxt
from pandas import read_csv

sequences, targets, groups, paths = load_dataset()
seq1 = [sequences[i] for i in range(len(groups)) if groups[i]==1]
seq2 = [sequences[i] for i in range(len(groups)) if groups[i]==2]
seq3 = [sequences[i] for i in range(len(groups)) if groups[i]==3]

targets1 = [targets[i] for i in range(len(groups)) if groups[i]==1]
targets2 = [targets[i] for i in range(len(groups)) if groups[i]==2]
targets3 = [targets[i] for i in range(len(groups)) if groups[i]==3]

es1 = create_dataset(seq1+seq2, targets1+targets2)
print('ES1: %s' % str(es1.shape))
savetxt('es1.csv', es1, delimiter=',')

es2_train = create_dataset(seq1+seq2, targets1+targets2)
es2_test = create_dataset(seq3, targets3)
print('ES2 Train: %s' % str(es2_train.shape))
print('ES2 Test: %s' % str(es2_test.shape))
savetxt('es2_train.csv', es2_train, delimiter=',')
savetxt('es2_test.csv', es2_test, delimiter=',')