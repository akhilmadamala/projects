import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft
import glob
from sklearn.neighbors import KNeighborsClassifier
from keras.preprocessing import sequence
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import svm
import termcolor
import pickle
import math
from sklearn.model_selection import KFold



def K_nearestneighbours(X_train, y_train, X_test, y_test):
    print("K-nearest neighbours")
    model = KNeighborsClassifier(n_neighbors=6)
    model.fit(X_train, y_train)
    filename = "KNN.pkl"
    pickle.dump(model, open(filename, 'wb'))
    resultPrintHelper("K-nearest neighbours", model, X_test, y_test)

# Fitting Naive Bayes to the Training set
def bayes_classifier(X_train, y_train, X_test, y_test):
    print("Bayes Started")
    bayesClassifier = GaussianNB()
    bayesClassifier.fit(X_train, y_train)
    resultPrintHelper("bayes_classifier", bayesClassifier, X_test, y_test)


# ## 2. Decision Tree Classifier

def decision_tree_classifier(X_train, y_train, X_test, y_test):
    print("DTC Started")

    dst_Classifier = DecisionTreeClassifier()
    dst_Classifier.fit(X_train, y_train)
    filename = "decision_tree.pkl"
    pickle.dump(dst_Classifier, open(filename, 'wb'))
    resultPrintHelper("decision_tree_classifier", dst_Classifier, X_test, y_test)


# ## 3. Random Forest Classifier

def random_forest_classifier(X_train, y_train, X_test, y_test):
    print("random forest  Started")
    rft_classifier = RandomForestClassifier(n_estimators=1000, n_jobs = -1, random_state=42)
    rft_classifier.fit(X_train, y_train)
    filename="Random_forest.pkl"
    pickle.dump(rft_classifier,open(filename, 'wb'))
    resultPrintHelper("random_forest_classifier", rft_classifier, X_test, y_test)


# ## 4. Logistic Regression Classifier

def logistic_regression_classifier(X_train, y_train, X_test, y_test):
    print("LogisticRegression Started")
    logistic_regression_classifier = linear_model.LogisticRegression(solver='lbfgs', max_iter=2000)
    logistic_regression_classifier.fit(X_train, y_train)
    filename = "logistic_regresion.pkl"
    pickle.dump(logistic_regression_classifier, open(filename, 'wb'))
    resultPrintHelper("logistic_regression_classifier", logistic_regression_classifier, X_test, y_test)


# ## 5. Support Vector Machine Classifier

def support_vector_machine_classifier(X_train, y_train, X_test, y_test):
    print("SVM Started")
    support_vector_machine_classifier = svm.SVC()
    support_vector_machine_classifier.fit(X_train, y_train)
    filename = "support_vector_machine_regresion.pkl"
    pickle.dump(support_vector_machine_classifier, open(filename, 'wb'))
    resultPrintHelper("support_vector_machine_classifier", support_vector_machine_classifier, X_test, y_test)


def resultPrintHelper(classifier_name, classifier, X_test, y_test1):
    # Predicting the Test set results
    d={1:0,2:0,3:0,4:0,5:0,6:0}

    print_st = termcolor.colored("Results of {0} classifier".format(classifier_name) , 'cyan', 'on_yellow', attrs=['bold', 'dark', 'underline', 'reverse'])
    print(print_st)
    y_pred1 = classifier.predict(X_test)
    ypred2=list(y_pred1)
    y_pred = []
    y_test=[]
    sum=0
    #print(len(y_pred),len(y_test))
    #print(y_pred)

    for i in range(len(y_pred1)//trunc_length):
        sum=0
        y=[]
        y1=[]
        for j in range(trunc_length*i,trunc_length*(i+1)-1):
            #print(y_pred[j])
            y.append(y_pred1[j])
            y1.append(y_test1[j])
        y_pred.append(max(y,key=y.count))
        y_test.append(max(y1, key=y.count))
    print(len(y_test))
    #print(y_pred)


        #value=max(d.items(), key=operator.itemgetter(1))[0]
        #ypred1.append(value)


    print("Accuracy score is: {}".format(accuracy_score(y_test, y_pred)))
    print("Precision score is: {}".format(precision_score(y_test, y_pred,average='micro')))
    print("Recall score is: {}".format(recall_score(y_test, y_pred,average='micro')))
    print("F1 score is: {}".format(f1_score(y_test, y_pred,average='micro')))
    print("------Confusion Matirx------")
    print(confusion_matrix(y_test, y_pred))




path=["C:\\Users\\akhil\\Desktop\\smaple data\\MC data\\buy\\BUY*.csv","C:\\Users\\akhil\\Desktop\\smaple data\\MC data\\communicate\\communicate*.csv"
,"C:\\Users\\akhil\\Desktop\\smaple data\\MC data\\Fun\\Fun*.csv","C:\\Users\\akhil\\Desktop\\smaple data\\MC data\\hope\\hope*.csv","C:\\Users\\akhil\\Desktop\\smaple data\\MC data\\mother\\mother*.csv"
,"C:\\Users\\akhil\\Desktop\\smaple data\\MC data\\really\\really*.csv"]

columns_to_be_read=[ 'nose_x', 'nose_y', 'leftEye_x', 'leftEye_y', 'rightEye_x', 'rightEye_y', 'leftEar_x', 'leftEar_y', 'rightEar_x', 'rightEar_y', 'leftShoulder_x', 'leftShoulder_y', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_x', 'leftElbow_y', 'rightElbow_x', 'rightElbow_y', 'leftWrist_x', 'leftWrist_y', 'rightWrist_x', 'rightWrist_y', 'leftHip_x', 'leftHip_y', 'rightHip_x', 'rightHip_y', 'leftKnee_x', 'leftKnee_y', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_x', 'leftAnkle_y', 'rightAnkle_x', 'rightAnkle_y']



data1=[]
counter=0
target=[]
k=1

for j in range(6):
    print(counter)
    counter=0
    for i in glob.glob(path[j]):
        fd = pd.read_csv(i)



        for i in columns_to_be_read:
            if i=="nose_x":
                continue
            if i=="nose_y":
                continue
            if "_x" in i:
                fd[i]=fd[i]-fd["nose_x"]
            if "_y" in i:
                fd[i]=fd[i]-fd["nose_y"]

        l_shoulder_x=fd['leftShoulder_x']-fd['rightShoulder_x']
        l_shoulder_y = fd['leftShoulder_y'] - fd['rightShoulder_y']
        shoulder_width=[]
        torso_length=[]
        for i in range(len(l_shoulder_x)):
            shoulder_width.append(math.sqrt(l_shoulder_x[i]**2+l_shoulder_y[i]**2))
        shoulder_width=pd.DataFrame(shoulder_width)

        hip_middle_x=(fd['leftHip_x']+fd['rightHip_x'])/2
        hip_middle_y=(fd['leftHip_y']+fd['rightHip_y'])/2

        torso_x=fd['nose_x']-hip_middle_x
        torso_y=fd['nose_y']-hip_middle_y

        for i in range(len(torso_x)):
            torso_length.append(math.sqrt(torso_x[i]**2+torso_y[i]**2))
        torso_length=pd.DataFrame(torso_length)


        for i in columns_to_be_read:
            if "_x" in i:
                fd[i]=fd[i]/shoulder_width
            if "_y" in i:
                fd[i]=fd[i]/torso_length




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
#plt.show()
#print(len_file)
print(pd.Series(len_file).describe())

no_columns=len(data1[0][0])

print("no of columns",no_columns)
#length decided=150
max_length=232
trunc_length=120

newdata=[]
#eveything padded to max length
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
for i in range(len(train1)):
    for j in range(len(train1[i])):
        print(train1[i][j][len(train1[0][0])-1])

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
