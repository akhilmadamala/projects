import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import spacy
import re
from spacy.lang.en.stop_words import STOP_WORDS
import os
#list of all stop words
stop_words = spacy.lang.en.stop_words.STOP_WORDS

'''
Description  : Searches and retrieves Job title, company name, Job description. (Helper funtion for general scraper)
input        : takes the beautiful soup parameter, Data holder dataframe and Functional area
output       : returns the data for the specific functional area  
'''
def scrapper(soup,df,area):
    # html tag remover
    cleaner = re.compile('<.*?>')
    # if any of the required files are not present then the code automatically gets what it needs
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        num = (len(df) + 1)
        job_post = []
        for a in div.find_all(name="a", attrs={"data-tn-element": "jobTitle"}):
            job_post.append(a["title"])
        company = div.find_all(name="span", attrs={"class": "company"})
        if len(company) > 0:
            for b in company:
                job_post.append(b.text.strip())
        else:
            sec_try = div.find_all(name="span", attrs={"class": "result-link-source"})
            for span in sec_try:
                job_post.append(span.text)

        for a in div.find_all(name="a", attrs={"data-tn-element": "jobTitle"}):
            link = a["href"]

            description = requests.get("https://www.indeed.com" + str(link)).text
            time.sleep(1)
            description = BeautifulSoup(description, "html.parser")

            for div in description.find_all(name="div", attrs={"class": "jobsearch-jobDescriptionText"}):
                # print(type(div))
                descp = re.sub(cleaner, "", str(div))
                # print(descp)
                job_post.append(descp)
            # job_post.append(description)
        if type(area) is tuple:
            job_post.append(area[1])
        else:
            job_post.append(area)
        print(job_post, len(job_post))
        if len(job_post) == 4:
            df.loc[num] = job_post
    return df

"""
Description  : master scraper for scraping data non company wise. it stores the retrived jobtitle, company name, 
               description and functional area in a csv file called scraped_data_with_descp.csv
input        : Null
output       : Null
"""
def genral_scraper(pages):
    maximum_posts = pages*10
    # html tag remover
    cleaner = re.compile('<.*?>')

    df = pd.DataFrame(columns=["job_title", "company_name", "description", "area"])
    count=1
    counter=0
    # funtional areas used for searching
    functional_area = ["Sales & Marketing", "Engineering Research & Development", "Customer Services",
                       "Business Operations", "Leadership", "Other"]

    # specific list of roles for ambiguous funtional areas observed while querying
    # add more roles to get better results
    leader_roles = ["supervisor", "visionary"]
    other = ["delivery"]
    for area in functional_area:
        print("no of queries retreive till now",len(df))
        for start in range(0, maximum_posts, 10):
            print("Results in page ", count)
            count+=1
            if area=='Leadership':
                for role in leader_roles:
                    counter+=1
                    if counter>maximum_posts:  #check if sum all posts mentioned in the titles doesn't exceed the maximum posts
                        continue
                    area_formatted = "+".join(role.split(" "))
                    source = requests.get('https://www.indeed.com/jobs?q=' + str(area_formatted) + '&start=' + str(start)).text
                    time.sleep(1)
                    soup = BeautifulSoup(source, "html.parser")
                    df=scrapper(soup,df,(area_formatted,area))
            elif area=='Other':
                counter=0
                for role in other:
                    if counter > maximum_posts:  # check if sum all posts mentioned in the titles doesn't exceed the maximum posts
                        continue
                    area_formatted = "+".join(role.split(" "))
                    source = requests.get('https://www.indeed.com/jobs?q=' + str(area_formatted) + '&start=' + str(start)).text
                    time.sleep(1)
                    soup = BeautifulSoup(source, "html.parser")
                    df = scrapper(soup, df, (area_formatted,area))
            else:
                area_formatted="+".join(area.split(" "))
                source = requests.get('https://www.indeed.com/jobs?q='+str(area_formatted)+'&start='+ str(start)).text
                time.sleep(2)
                soup = BeautifulSoup(source, "html.parser")
                df = scrapper(soup, df, area)

    df.to_csv("scraped_data_with_descp.csv",index=False)## changed
    print("sample data present in the dataframe")

    print(df.head())


'''
Description: scrapes data company wise after preprocessing by removing punctuation, stopwords, numerical values.
             it gives necessary suggestions necessary modifications and also tells when a result is excluded when it 
             doesn't meet certain heuristics
             hueristics:
                       1. when the all keywords used for searching are not found in the company name
                          issues a statements saying a certain result is excluded
                       2. if no results founds for a particular company 
                          isses a statement to modify the company name
                       3. Also a changable filter is found on line 136 which filters the comapany names.       
input      : Takes no of pages to retreive results from after the search
output     : writes out a csv file containing the results wih column names company name, job title and description 
'''
def company_wise_scrapper(pages):
    df = pd.DataFrame(columns=["company_name", "job_title", "description"])
    maximum_posts = pages*10
    companies = []
    # html tag remover
    cleaner = re.compile('<.*?>')

    with open(os.getcwd()+"/resources/enlyft_datascience_sample.csv") as f:
        # with open("sample.csv") as f:
        for line in f:
            if line.lower() != "company name\n":
                punctuations = '''!()-[]{};:'"\,<>./?@#$%^*_~''' #& symbol is removed because some company name have it and removing those results in empty results
                line = "".join([s for s in line if s not in punctuations and s not in "123456789"])
                line = line.strip("\n").strip('&').lower().split(" ")
                #excluding Business terms in the comapany names
                line = [word for word in line if
                        len(word) != 0 and word not in ["corporation", "us", "science", "technologies", "holdings",
                                                        "corp", "llc", "inc", 'consulting', "international", "holdings",
                                                        "ltd", "co", "plc", "holding", 'association', "se", "ag",
                                                        "consultancy", "Incorporated", "financial", "services", "llp",
                                                        "oyj"] and word not in stop_words]

                if "".join(line).endswith("com"):
                    companies.append(" ".join(line)[:-len("com")])
                else:
                    companies.append(" ".join(line))


    print("Total no of companies present in the list", len(companies))
    companies = list(set(companies))
    print("No of unique companies in the list ", len(companies))

    print("modified list of companies")
    for company in companies:
        if len(company) > 0:
            print(company)

    print("===========================================================")

    for company_name in companies:
        if len(company_name) > 0:
            for start in range(0, maximum_posts, 10):
                if '&' not in company_name:
                    # print("hello")
                    company_ = company_name.split(" ")
                    source = requests.get('http://indeed.com/jobs?q=company%3A+' + str(
                        "+".join(company_name.split(" "))) + '&start=' + str(start)).text
                else:
                    l = "+".join(company_name.split(" "))
                    index = l.index('&')
                    company_ = company_name.split(" ")
                    print('http://indeed.com/jobs?q=company%3A+' + str(l[:index]) + '%26' + str(
                        l[:index]) + '&start=' + str(start))
                    source = requests.get('http://indeed.com/jobs?q=company%3A+' + str(l[:index]) + "%26" + str(
                        l[index + 1:]) + '&start=' + str(start)).text

                print("Key words used to search the Job posting website", company_)

                print("http://indeed.com/jobs?q=company%3A" + str(
                    "+".join(company_name.split(" "))).lower() + '&start=' + str(start))
                # time.sleep(1)
                soup = BeautifulSoup(source, "html.parser")
                # print(soup)
                if "row result" in source:
                    for div in soup.find_all(name="div", attrs={"class": "row"}):
                        num = (len(df) + 1)
                        job_post = []
                        company = div.find_all(name="span", attrs={"class": "company"})

                        if len(company) > 0:
                            for b in company:
                                company_ = [word for word in company_ if word]
                                if len(set(company_).difference(set(b.text.strip().lower().split(" ")))) == 0:
                                    job_post.append(b.text.strip())
                                    for a in div.find_all(name="a", attrs={"data-tn-element": "jobTitle"}):
                                        job_post.append(a["title"])

                                    for a in div.find_all(name="a", attrs={"data-tn-element": "jobTitle"}):
                                        link = a["href"]

                                        description = requests.get("https://www.indeed.com" + str(link)).text
                                        description = BeautifulSoup(description, "html.parser")

                                        for div in description.find_all(name="div", attrs={
                                            "class": "jobsearch-jobDescriptionText"}):
                                            descp = re.sub(cleaner, "", str(div))
                                            job_post.append(descp)
                                    print(job_post, len(job_post))
                                    if len(job_post) == 3:
                                        df.loc[num] = job_post
                                else:
                                    print("found a partial matching company for {} but excluded try modifying the company name to include it".format(company_name))
                else:
                    print("no results found on page {}".format(start % 10))
                    print(
                        "Check the company name {} for mistakes if not try abbreviations of the company other reasons include no job posting on the websitr".format(
                            company_name))

                    break
    df.to_csv("scraped_data_company_wise.csv", index=False)
    print("sample data present in the dataframe")
    print(df.head())

