from pandas import read_csv
from os import listdir
from numpy import array

# return list of traces, and arrays for targets, groups and paths
def load_dataset(prefix=''):
    grps_dir, data_dir = prefix + 'groups/', prefix + 'dataset/'
    # load mapping files
    targets = read_csv(data_dir + 'MovementAAL_target.csv', header=0)
    groups = read_csv(grps_dir + 'MovementAAL_DatasetGroup.csv', header=0)
    paths = read_csv(grps_dir + 'MovementAAL_Paths.csv', header=0)
    # load traces
    sequences = list()
    target_mapping = None
    for name in listdir(data_dir):
        filename = data_dir + name
        if filename.endswith('_target.csv'):
            continue
        df = read_csv(filename, header=0)
        values = df.values
        sequences.append(values)
    return sequences, targets.values[:, 1], groups.values[:, 1], paths.values[:, 1]

def create_dataset(sequences, targets):

	transformed = list()
	n_vars = 4
	n_steps = 19

	for i in range(len(sequences)):
		seq = sequences[i]
		vector = list()
		# last n observations
		for row in range(1, n_steps+1):
			for col in range(n_vars):
				vector.append(seq[-row, col])

		vector.append(targets[i])

		transformed.append(vector)

	transformed = array(transformed)
	transformed = transformed.astype('float32')
	return transformed