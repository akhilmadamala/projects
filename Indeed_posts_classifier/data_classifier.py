from  model_final import predictors,spacy_tokenizer,tfidf_vector,save_model,algorithm_spot_check,grid_searcher
import pickle
import pandas as pd
from collections import defaultdict
import os
from os import path
from scrapper import company_wise_scrapper,genral_scraper
'''
Description: takes a dataframe of descriptions and predicts the functional area. Also writes out a csv file
             containing the predictions concatinated to the passed dataframe
input      : takes a dataframe containing column names company_name,job_title,description
output     : returns a result dataframe where the predictions are concatinated with the column name 'area' to the passed
             input 
'''
def functional_area_predictor(data):
    dir=os.getcwd()
    model = pickle.load(open(dir+'\\models\\final_model_svm.sav', 'rb'))
    areas = pd.DataFrame(model.predict(data['description']), columns=["area"])
    result = pd.concat([data, areas], axis=1)
    result['company_name'] = result['company_name'].str.lower()
    result.to_csv("results.csv", index=False)
    return result

'''
Description: takes a dataframe and counts the tags by company name and writes out a csv file called final_result.csv 
             containing the company wise count of jobs per functional areas and total no of jobs 
input      : takes a dataframe containing column names company_name,job_title,description,area
output     : returns a company wise count of jobs per functional areas and total no of jobs 
'''
def jobs_counter(result):
    functional_area = {"Sales & Marketing": 0, "Engineering Research & Development": 1, "Customer Services": 2,
                       "Business Operations": 3, "Leadership": 4, "Other": 5}

    d = defaultdict(list)

    for i in range(len(result)):
        if result.iloc[i, 0] in d:
            d[str(result.iloc[i, 0])][functional_area[result.iloc[i, 3]]] += 1
        else:
            d[str(result.iloc[i, 0])] = [0, 0, 0, 0, 0, 0]



    counter = pd.DataFrame.from_dict(d, orient='index',
                                     columns=["Sales & Marketing", "Engineering Research & Development",
                                              "Customer Services", "Business Operations", "Leadership", "Other"])
    counter['total_jobs'] = counter[list(counter.columns)].sum(axis=1)
    counter.to_csv("final_result.csv")
    return counter

if __name__ == '__main__':

    if not path.exists(os.getcwd()+"/resources/scraped_data_with_descp.csv"):
        print("scrapping training data")
        genral_scraper(50)         #here 50 indicates no of pages to query for results
    if not path.exists(os.getcwd() + "/resources/scraped_data_company_wise.csv"):
        #print('hello')
        print("scrapping Company wise data for prediction")
        company_wise_scrapper(50)  #here 50 indicates no of pages to query for results

    # to run algorithm spot check load training data
    #un comment below line for spot check results
    #algorithm_spot_check(pd.read_csv(os.getcwd()+"/resources/scraped_data_with_descp.csv"))


    if os.path.exists(os.getcwd()+'/models/') and os.path.isdir(os.getcwd()+'/models/'):
        if not os.listdir(os.getcwd()+'/models/'):
            print("training the model using naive bayes")
            df_data = pd.read_csv(os.getcwd()+"/resources/scraped_data_with_descp.csv")

            print("===========Description data  used for training================")

            print("============Distribution of data===========")
            print(df_data.area.value_counts())
            save_model(df_data)

    print("required files and model found")
    df=pd.read_csv(os.getcwd()+"/resources/scraped_data_company_wise.csv")
    X=df["description"]
    result=functional_area_predictor(df)
    print(result.head())
    final_result=jobs_counter(result)
    print(final_result.head())






