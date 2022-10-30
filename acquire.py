# dependencies
import os
import pandas as pd
import numpy as np

# key visualization libraries/modules
import matplotlib.pyplot as plt
import seaborn as sns

# JSON import
import json





# -------- # 

def get_reddit_stress():
    '''Function to import the initial Reddit Stress Data.
    
    1. check if data already exists locally as a .csv file
    2. if yes --> return .csv file/data as a Pandas dataframe
    3. if no --> use the git url to import and clean the original data
    4. if no cont. --> cache the stress data as a .csv file for future referencing
    5. if no cont. --> return called data as a Pandas dataframe
    '''
    
    # cached filename to look for
    filename = "stress.csv"

    # search for filename in local/OS directory
    if os.path.isfile(filename):

        # if file and filename exists, then return csv as Pandas df
        # future iteration: consider using relative path 
        # this may help to prevent referencing deleted/incorrect files
        df = pd.read_csv(filename, index_col = "social_timestamp")

        # print df shape
        print(f'dataframe shape: {df.shape}')

        # return the dataframe
        return df

    # if file/csv can't be found in OS directory, then access the data and cache it as a csv locally
    else:

        url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/stress.csv"
        # read text data as csv and convert to pandas dataframe
        df = pd.read_csv(url)

        # let's move forward with just the following columns/features
        df = df[[
            "label", 
            "post_id",
            "subreddit",                                 
            "sentence_range",              
            "text",                        
            "id",              
            "confidence",            
            "social_timestamp",           
            "social_karma",                
            "syntax_ari",
            "sentiment",
            "social_upvote_ratio",
            "social_num_comments"
            ]]

        # timestamp appears to be in "epoch seconds format"
        df["social_timestamp"] = pd.to_datetime(df['social_timestamp'], unit = 's')

        # sort and set data as index
        df.set_index('social_timestamp', inplace = True)

        # cache the data for easier/quicker reference
        df.to_csv("stress.csv")

        # print the shape
        print(f'dataframe shape: {df.shape}')

        # return the dataframe
        return df

