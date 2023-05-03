"""
Code used in task 1 and 2 to load and process datasets from files
For task 2: loads the dataset and stores the relevant features in temporary files
Extracts and cleans the text of reviews as well as the review ratings
Splits the data 80/20 into training/testing
"""
import json
import re
from string import punctuation

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
# nltk.download('stopwords') - only need to run once on a machine
# nltk.download('punkt') - only need to run once on a machine

stop_words = set(stopwords.words('english'))    

def clean_text(text):
    """Cleans a text by removing links, punctuation, and stop words.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    # Remove links
    text = re.sub(r'http\S+', '', text)

    # Remove punctuation characters
    text = text.translate(str.maketrans('', '', punctuation))

    # Tokenize the words
    words = word_tokenize(text)

    # Remove stop words
    words = [word for word in words if word.lower() not in stop_words]

    # Join the words into a sentence
    text = ' '.join(words)

    return text

def load_review_dataset(fname='/Users/peterhowell/Downloads/yelp_dataset/yelp_academic_dataset_review.json') -> pd.DataFrame:
    """Load the review json file at `fname` into a dataframe
    
    Args:
        fname (str): name of review json file
    Returns:
        pd.DataFrame: dataframe of reviews
    """
    # Loading the review data from json file 
    rev_data = []
    with open(fname, 'r') as f:
        for line in f:
            rev_data.append(json.loads(line))

    review_df = pd.DataFrame(rev_data)
    del rev_data
    print('review dataframe loaded')
    return review_df

# Task 1
def load_business_dataset(fname='business.json') -> dict:
    """loads data stored in file `fname` and returns list of businesses and their attributes

    Args:
        fname (str, optional): file name. Defaults to 'business.json'.

    Returns:
        dict: dictionary with business_id as keys and attributes as values
    """
    bus_data = {}
    with open(fname, 'r') as f:
        for line in f:
            json_line = json.loads(line.strip())
            attrs = json_line['attributes']
            
            # only include businesses that have at least one attribute
            if  attrs is not None:
                bus_attrs = {}
                # check if each attribute is a boolean
                for attr, value in attrs.items():
                    # check for string storage of boolean
                    if isinstance(value, str) and value.lower() in ["true", "false"]:
                        value = value.lower() == "true"
                    if isinstance(value, bool):
                         # store boolean as int, 0 or 1
                        bus_attrs[attr] = int(value)
                    if bus_attrs:
                        bus_data[json_line['business_id']] = bus_attrs
    print('business dataset loaded')
    return bus_data

def clean_reviews(review_df: pd.DataFrame) -> pd.DataFrame:
    """applies `clean_text` to the reviews the dataframe

    Args:
        reviews (pd.DataFrame): reviews to clean

    Returns:
        pd.DataFrame: dataframe with the reviews cleaned
    """
    print('beginning to clean reviews')
    review_df['text'] = review_df['text'].apply(clean_text)
    print('reviews cleaned')
    return review_df

def process_reviews(review_df: pd.DataFrame) -> pd.DataFrame:
    """Processes reviews for use in task 2 by removing columns, cleaning the text, etc

    Args:
        review_df (pd.DataFrame): Review dataframe

    Returns:
        pd.DataFrame: processed review dataframe
    """
    review_df.drop(['review_id','user_id',	'business_id','useful','funny','cool', 'date'], axis=1, inplace=True)

    review_df['stars'] = review_df['stars'].astype(int)
    # ## Identifying and extracting target

    # currently using:
    # convert star rating to:   negative, neutral,  positive
    #                           [1,0,0]   [0,1,0]   [0,0,1]
    review_df['negative'] = (review_df['stars'] <  3).astype(int)
    review_df['neutral']  = (review_df['stars'] == 3).astype(int)
    review_df['positive'] = (review_df['stars'] >  3).astype(int)

    review_df.drop(['stars'], axis=1, inplace=True)

    review_df = clean_reviews(review_df)
    return review_df

def dump_reviews(df: pd.DataFrame):
    """splits dumps dataset into files for later usage

    Args:
        df (pd.DataFrame): dataframe
    """
    train, test = train_test_split(df, train_size=0.8, random_state=42)
    print('train shape: ', train.shape)
    print('test shape:', test.shape)

    train[['negative', 'neutral', 'positive']].to_pickle('data/train_labels')
    train[['text']].to_pickle('data/train_text')
    print('train data saved')

    test[['negative', 'neutral', 'positive']].to_pickle('data/test_labels')
    test[['text']].to_pickle('data/test_text')
    print('test data saved')

if __name__ == '__main__':
    # Task 2
    review_df = load_review_dataset()
    review_df = process_reviews(review_df)
    dump_reviews(review_df)
