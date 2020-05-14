import string
import numpy as np
import pandas as pd
import spacy
from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from spacy.lang.en import English
import pickle
import os
'''
Description: Takes a sentence and preprocesses it by doing lemmitization, removing stopping words, punctuations and 
             numerical values
input      : Takes a sentence of type string
output     : Returns tokens of the sentence which may be used as features for the model  
'''
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    parser = English()
    punctuations = string.punctuation
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]
    mytokens=[word for word in mytokens if not any(c.isdigit() for c in word)]


    return mytokens

'''
Description: converts a text into lower and strips spaces which prevents tokenizing empty strings
             inherited from TransformerMixin class to inherit fit and transform since they are needed when out in a 
             pipeline
Input      : takes a dataframe containing column names company_name,job_title,description
Output     : returns a result dataframe where the predictions are concatinated with the column name 'area' to the passed
             input 
'''
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [text.strip().lower() for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 1))


'''
Description: prints out accuracy, recall and precision values for the classifier of interest
input      : takes a name of classifier, test_data and predicted data by the model
output     : prints out train and validation scores 
'''

def scores(classifer_name,test_data,predicted_data):
    print("{} Accuracy:{}".format(classifer_name,metrics.accuracy_score(test_data, predicted_data)))
    print("{} Precision:{}".format(classifer_name,metrics.precision_score(test_data, predicted_data, average='micro')))
    print("{} Recall: {}".format(classifer_name,metrics.recall_score(test_data, predicted_data, average='micro')))


'''
Description: initial analysis phase to determine the best algorithm and mesure performance
input      : takes a dataframe containing column names company_name,job_title,description
output     : prints out train and validation scores 
'''

def algorithm_spot_check(df_data):
    print("spot checking using naive bayes, logistic regression, svm and random forest")
    #df_data=pd.read_csv(os.getcwd()+"/resources/scraped_data_with_descp.csv")

    X = df_data['description']
    ylabels = df_data['area']

    X_train, X_test, y_train, y_test = train_test_split(X.values, ylabels.values, test_size=0.3)
    classifier1 = MultinomialNB()
    classifier2 = LogisticRegression()
    classifier3 = SGDClassifier()
    classifier4 = RandomForestClassifier()


    pipe1 = Pipeline([("cleaner", predictors()), ('vectorizer', tfidf_vector), ('classifier', classifier1)])
    pipe2 = Pipeline([("cleaner", predictors()), ('vectorizer', tfidf_vector), ('classifier', classifier2)])
    pipe3 = Pipeline([("cleaner", predictors()), ('vectorizer', tfidf_vector), ('classifier', classifier3)])
    pipe4 = Pipeline([("cleaner", predictors()), ('vectorizer', tfidf_vector), ('classifier', classifier4)])

    print("=======================================================")


    pipe1.fit(X_train, y_train)
    tdf = pipe1.predict(X_train)

    print("=============training Phase==========================")
    print("training accuracy", np.mean(tdf == y_train))

    predicted = pipe1.predict(X_test)
    #scores('Naive Bayes',X_test,predicted)
    #print(" Accuracy:{}".format( metrics.accuracy_score(y_test, predicted)))
    #print("Precision:{}".format( metrics.precision_score(y_test, predicted, average='micro')))
    #print("Recall: {}".format( metrics.recall_score(Y_test, predicted, average='micro')))
    #print("=========Classificaion report==========\n", metrics.classification_report(X_test, predicted))
    scores("naive bayes",y_test,predicted)
    print("=======================================================")

    pipe2.fit(X_train, y_train)
    tdf = pipe2.predict(X_train)
    print("=============trianing Phase==========================")
    print("training accuracy", np.mean(tdf == y_train))
    print("=============testing Phase============================")
    # Predicting with a test dataset
    predicted = pipe2.predict(X_test)
    scores("logistic regression",y_test,predicted)
    print("=======================================================")

    pipe3.fit(X_train, y_train)
    tdf = pipe3.predict(X_train)
    print("=============training Phase==========================")
    print("training accuracy", np.mean(tdf == y_train))

    # Predicting with a test dataset
    print("=============testing Phase============================")
    predicted = pipe3.predict(X_test)

    scores("svm",y_test,predicted)

    print("=======================================================")

    pipe4.fit(X_train, y_train)
    tdf = pipe4.predict(X_train)
    print("=============training Phase==========================")
    print("trianing accuracy", np.mean(tdf == y_train))
    # Predicting with a test dataset
    print("=============testing Phase============================")
    predicted = pipe4.predict(X_test)
    scores("random forest",y_test,predicted)


'''
Description: does grid search for the four classifiers used naive bayes, Logistic regression, Svm, and random forest
input      : takes a dataframe containing column names company_name,job_title,description,area
output     : prints out extensive report of best parameters and different accuracy, recall, f1 and support for each 
             cross validation.
'''
def grid_searcher(df_data):
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    X,Y =df_data["description"].values, df_data["area"].values
    classifier1 = MultinomialNB()
    classifier2 = LogisticRegression()
    classifier3 = SGDClassifier()
    classifier4 = RandomForestClassifier()

    print("After grid search")
    pipe1 = Pipeline([("cleaner", predictors()),('vectorizer', tfidf_vector),('classifier', classifier1)])
    pipe2 = Pipeline([("cleaner", predictors()),('vectorizer', tfidf_vector),('classifier', classifier2)])
    pipe3 = Pipeline([("cleaner", predictors()),('vectorizer', tfidf_vector),('classifier', classifier3)])
    pipe4= Pipeline([("cleaner", predictors()),('vectorizer', tfidf_vector),('classifier', classifier4)])

    parameters1 = {'vectorizer__ngram_range': [(1, 1), (1, 2)],'classifier__alpha': (1e-2, 1e-3),}
    parameters2 = {'vectorizer__ngram_range': [(1, 1), (1, 2)],'classifier__max_iter':(100,200,300),}
    parameters3 = {'vectorizer__ngram_range': [(1, 1), (1, 2)],'classifier__alpha': (1e-2, 1e-3,1e-4),}
    parameters4 = {'vectorizer__ngram_range': [(1, 1), (1, 2)],}
    print("==============Naive bayes=======================")
    gs1=GridSearchCV(pipe1, parameters1,cv=5)
    gs1=gs1.fit(X,Y)
    df1=pd.DataFrame(gs1.cv_results_)
    print(gs1.best_score_,gs1.best_params_)
    print("========summary of results =========")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df1)
    print("===================================================================")
    
    print("==============logistic regression=======================")

    gs2=GridSearchCV(pipe2, parameters2,cv=5)
    gs2=gs2.fit(X,Y)
    df2=pd.DataFrame(gs2.cv_results_)
    print(gs2.best_score_,gs2.best_params_)
    print("========summary of results=========")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df2)
    

    print("===================================================================")
    print("==============SVM=======================")

    gs3=GridSearchCV(pipe3, parameters3,cv=5)
    gs3=gs3.fit(X,Y)
    print(gs3.best_score_,gs3.best_params_)
    print("========summary of results=========")
    df3=pd.DataFrame(gs3.cv_results_)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df3)

    print("===================================================================")
    print("==============Random forest=======================")

    gs4=GridSearchCV(pipe4, parameters4,cv=5)
    gs4=gs4.fit(X,Y)
    print(gs4.best_score_,gs4.best_params_)
    print("========summary of results=========")
    df4=pd.DataFrame(gs4.cv_results_)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df4)




'''
Description: takes a dataframe containing column names company_name,job_title,description,area and trains a model and 
             saves it
input      : takes a dataframe containing column names company_name,job_title,description,area
output     : saves the model in models folder by name final_model_svm.sav 
'''
def save_model(input):
    X,Y =input["description"].values, input["area"].values
    classifier3 = SGDClassifier(alpha=0.001)
    pipe3 = Pipeline([("cleaner", predictors()), ('vectorizer', tfidf_vector), ('classifier', classifier3)])
    pipe3.fit(X,Y)
    filename = 'final_model_svm.sav'
    pickle.dump(pipe3, open(os.getcwd()+'/models/'+filename, 'wb'))

