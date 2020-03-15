from pandas import read_csv
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

train = read_csv('es2_train.csv', header=None)
test = read_csv('es2_test.csv', header=None)

trainX, trainy = train.values[:, :-1], train.values[:, -1]
testX, testy = test.values[:, :-1], test.values[:, -1]

models, names = list(), list()

models.append(LogisticRegression())
names.append('LR')

models.append(KNeighborsClassifier())
names.append('KNN')

models.append(KNeighborsClassifier(n_neighbors=7))
names.append('KNN-7')

models.append(DecisionTreeClassifier())
names.append('CART')

models.append(SVC())
names.append('SVM')

models.append(RandomForestClassifier())
names.append('RF')

models.append(GradientBoostingClassifier())
names.append('GBM')

all_scores = list()
for i in range(len(models)):
	scaler = StandardScaler()
	model = Pipeline(steps=[('s',scaler), ('m',models[i])])
	model.fit(trainX, trainy)
	yhat = model.predict(testX)
	score = accuracy_score(testy, yhat) * 100
	all_scores.append(score)
	print('%s %.3f%%' % (names[i], score))

pyplot.bar(names, all_scores)
pyplot.show()