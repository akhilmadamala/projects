import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft
import glob
from sklearn.neighbors import KNeighborsClassifier
from keras.preprocessing import sequence
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, precision_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
import termcolor
import pickle
from sklearn.model_selection import KFold



def K_nearestneighbours(X_train, y_train, X_test, y_test):
    print("K-nearest neighbours")
    model = KNeighborsClassifier(n_neighbors=6)
    model.fit(X_train, y_train)
    filename = "KNN.pkl"
    pickle.dump(model, open(filename, 'wb'))
    accuracy_measurer("K-nearest neighbours", model, X_test, y_test)

# Fitting Naive Bayes to the Training set
def bayes_classifier(X_train, y_train, X_test, y_test):
    print("Bayes Started")
    bayesClassifier = GaussianNB()
    bayesClassifier.fit(X_train, y_train)
    accuracy_measurer("bayes_classifier", bayesClassifier, X_test, y_test)


# ## 2. Decision Tree Classifier

def decision_tree_classifier(X_train, y_train, X_test, y_test):
    print("Decision tree started")

    dst_Classifier = DecisionTreeClassifier()
    dst_Classifier.fit(X_train, y_train)
    filename = "decision_tree.pkl"
    pickle.dump(dst_Classifier, open(filename, 'wb'))
    accuracy_measurer("decision_tree_classifier", dst_Classifier, X_test, y_test)


# ## 3. Random Forest Classifier

def random_forest_classifier(X_train, y_train, X_test, y_test):
    print("Random Forest Classifier Started")
    rft_classifier = RandomForestClassifier(n_estimators=1000, n_jobs = -1, random_state=42)
    rft_classifier.fit(X_train, y_train)
    filename="Random_forest.pkl"
    pickle.dump(rft_classifier,open(filename, 'wb'))
    accuracy_measurer("random_forest_classifier", rft_classifier, X_test, y_test)


# ## 4. Logistic Regression Classifier

def logistic_regression_classifier(X_train, y_train, X_test, y_test):
    print("Logistic regression  Started")
    logistic_regression_classifier = linear_model.LogisticRegression(solver='lbfgs', max_iter=2000)
    logistic_regression_classifier.fit(X_train, y_train)
    filename = "logistic_regresion.pkl"
    pickle.dump(logistic_regression_classifier, open(filename, 'wb'))
    accuracy_measurer("logistic_regression_classifier", logistic_regression_classifier, X_test, y_test)






def accuracy_measurer(classifier_name, classifier, X_test, y_test1):
    # Predicting the Test set results
    print_st = termcolor.colored("Results of {0} classifier".format(classifier_name) , 'cyan', 'on_yellow', attrs=['bold', 'dark', 'underline', 'reverse'])
    print(print_st)
    y_pred1 = classifier.predict(X_test)
    ypred2=list(y_pred1)
    y_pred = []
    y_test=[]

    for i in range(len(y_pred1)//trunc_length):

        y=[]
        y1=[]
        for j in range(trunc_length*i,trunc_length*(i+1)-1):

            y.append(y_pred1[j])
            y1.append(y_test1[j])
        y_pred.append(max(y,key=y.count))
        y_test.append(max(y1, key=y.count))
    print(len(y_test))

    print("Accuracy score is: {}".format(accuracy_score(y_test, y_pred)))
    print("Precision score is: {}".format(precision_score(y_test, y_pred,average='micro')))
    print("Recall score is: {}".format(recall_score(y_test, y_pred,average='micro')))
    print("F1 score is: {}".format(f1_score(y_test, y_pred,average='micro')))
    print("------Confusion Matirx------")
    print(confusion_matrix(y_test, y_pred))


path=["C:\\Users\\akhil\\Desktop\\smaple data\\MC data\\buy\\BUY*.csv","C:\\Users\\akhil\\Desktop\\smaple data\\MC data\\communicate\\communicate*.csv"
,"C:\\Users\\akhil\\Desktop\\smaple data\\MC data\\Fun\\Fun*.csv","C:\\Users\\akhil\\Desktop\\smaple data\\MC data\\hope\\hope*.csv","C:\\Users\\akhil\\Desktop\\smaple data\\MC data\\mother\\mother*.csv"
,"C:\\Users\\akhil\\Desktop\\smaple data\\MC data\\really\\really*.csv"]

data1=[]
counter=0
target=[]
k=1

for j in range(6):
    print(counter)
    counter=0
    for i in glob.glob(path[j]):
        fd = pd.read_csv(i)
        fd=rfft(fd)
        fd=pd.DataFrame(fd)
        fd['label']=[k]*fd.shape[0]
        df = fd.values
        data1.append(df)
        counter+=1
    k+=1




#making the csv files to same length

len_file=[]

for i in range(len(data1)):
    len_file.append(len(data1[i]))
plt.hist(len_file,5)
print(pd.Series(len_file).describe())

no_columns=len(data1[0][0])

print("no of columns",no_columns)

max_length=232
trunc_length=120

newdata=[]
for i in data1:
    diff=max_length-len(i)
    padding=np.repeat(i[-1],diff).reshape(no_columns,diff).transpose()
    #print(i[-1])
    newdata.append(np.concatenate([i,padding]))


train1=sequence.pad_sequences(newdata,trunc_length,padding='post', dtype='float', truncating='post')


number_of_splits = 5
kf = KFold(n_splits = number_of_splits, shuffle = True, random_state = 2)
t=[]
ts=[]
training_set=[]
testing_set=[]
train1=np.asarray(train1)

for fold, (train_index, test_index) in enumerate(kf.split(train1)):
    print("present fold {0}".format(fold))

    training_set=[]
    testing_set=[]
    target1=[]
    target2=[]
    print(len(test_index))
    for i in train_index:
        training_set.extend(train1[i])
        target1.extend(train1[i, :, len(train1[0][0]) - 1])

    for j in test_index:
        testing_set.extend(train1[j])
        target2.extend(train1[j, :, len(train1[0][0]) - 1])
    training_set=np.asarray(training_set)
    training_set=np.delete(training_set,no_columns-1,1)
    training_set=list(training_set)
    testing_set = np.asarray(testing_set)
    testing_set = np.delete(testing_set,no_columns-1, 1)
    testing_set = list(testing_set)

    random_forest_classifier(training_set, target1, testing_set, target2)
    decision_tree_classifier(training_set, target1, testing_set, target2)
    logistic_regression_classifier(training_set, target1, testing_set, target2)
    K_nearestneighbours(training_set, target1, testing_set, target2)



