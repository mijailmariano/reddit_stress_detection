# notebook dependencies
import os 
import pandas as pd
import numpy as np

# regular expression import
import re

# uni-code library
import unicodedata

# natural language toolkit library/modules
import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer

from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

# sklearn library/modules
from sklearn.model_selection import train_test_split


# lowercases all letters
# normalizes unicode characters
# replaces non-alphanumeric characters with whitespace
def basic_clean(string):
    '''Key text cleaning functions'''

    # lowercase the text
    string = string.lower()

    # normalizing the text
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # return only alphanumeric values in text: everything else, convert to whitespace
    string = re.sub("[^a-z0-9\s']", ' ', string)

    # cleans multi-line strings in the data
    string = re.sub(r"[\r|\n|\r\n]+", ' ', string)

    # removing any word/ele <= 2 letters
    string = re.sub(r'\b[a-z]{,2}\b', '', string)
    
    # removing multiple spaces
    string = re.sub(r'\s+', ' ', string)

    # removing beginning and end whitespaces
    string = string.strip()

    # return the string text
    return string


def tokenize(string):
    '''Function that tokenizes the string text'''

    # creating the tokenize object
    tokenizer = ToktokTokenizer()
    
    # using the tokenize object on the input string
    return tokenizer.tokenize(string, return_str = True)



def porter_stem(string):
    '''Function that uses the "PorterStem" method on the text data'''

    # creating the object
    ps = PorterStemmer()
    
    # using list comprehension to return the stem of ea. word in the string as a list
    stems = [ps.stem(word) for word in string.split()]

    # then re-joining ea. word as a single string text w/ a space in between ea. word
    stemmed_string = ' '.join(stems)

    return stemmed_string



def lemmatize(string):
    '''Function to lemmatize text'''

    # creating the lemmatizer object
    wnl = WordNetLemmatizer()
    
    # using list comprehension to apply the lemmatizer on ea. word and return words as a list
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    
    # re-joining the individual words as a single string text
    lemmatized_string = ' '.join(lemmas)
    
    # return the tranformed string text
    return lemmatized_string



def remove_stopwords(string, exclude_words = None, include_words = None):
    '''Function that removes stop words in text'''

    # including potential redundant words in scrape
    include_words = [
                    "metaverse", 
                    "Metaverse", 
                    "meta verse", 
                    "Meta Verse", 
                    "Meta verse",
                    'see', 'http', 'com','github','org', 'source', 'example', 'code', 'use', '1', "'", ';', '&#9']

    # creating the list of english stop words
    stopword_list = stopwords.words('english')
    
    # if there are words to exlude not in stopword_list, then add them to stop word list
    if include_words:
        
        stopword_list = stopword_list + include_words

    # if there are words we dont want to remove, then take them out of the stop words list
    if exclude_words:
        
        for word in exclude_words:
            
            stopword_list.remove(word)

    # split string text into individual words        
    words = string.split()
    
    # filter the string words, and only include words not in stop words list
    filtered_words = [word for word in words if word not in stopword_list]
    
    # re-join the words into individual string text
    filtered_string = ' '.join(filtered_words)
    
    # return the string text back: excluding stop words
    return filtered_string


def clean_data_objects(df):
    '''Function to clean the original df data types'''

    df = df[[
        "repo", \
        "language", \
        "readme_contents"]].astype(str)

    print(f'df shape: {df.shape}')

    return df


def mass_text_clean(text, include_words=None, exclude_words=None):
    '''Function to mass dataclean the original README repo files'''

    text = basic_clean(text)

    text = lemmatize(text)

    text = remove_stopwords(text, include_words = include_words, exclude_words = exclude_words)

    return text


def update_languages(df):
    ''' Dataset has multiple languages. Some languages are iterations of other languages.
    This function is to replace some of the languages names so the are in the same category'''

    # Jupyter notebook as Python
    df = df.replace('Jupyter Notebook', 'Python')

    # C Related Languages
    df = df.replace(('Objective-C++','Objective-C','C','C++','C#','SuperCollider','GLSL'),'C')

    # Java Related Languages ans sublanguages
    df = df.replace(('JavaScript', 'Vue','Clojure','Kotlin','EJS'), 'Java')

    # TypeScript Languages
    df = df.replace('Svelte','TypeScript')

    # Microsoft languages for windows applications
    df = df.replace(('PowerShell','Batchfile','Shell'),'Microsoft')

    # iOS related languages
    df = df.replace(('Metal','Swift'),'iOS')

    # languages with < 10 usage
    df = df.replace(('Microsoft','ShaderLab','SCSS','PHP','iOS','GDScript','Roff','HCL','TeX', 'Cadence','R','LSL','ASP.NET'),'other')

    # NaNs
    df=df.replace('nan', 'text')
    return df



def train_validate_test_split(df, target, seed = 123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''

    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])

    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])


    # printing the shapes of the datasets
    print(f'train shape: {train.shape}')
    print(f'validate shape: {validate.shape}')
    print(f'test shape: {test.shape}')

    return train, validate, test