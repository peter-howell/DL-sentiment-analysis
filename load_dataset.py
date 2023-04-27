"""
Code that loads the dataset and stores the relevant features in temporary files
Extracts and cleans the text of reviews as well as the review ratings
Splits the data 80/20 into training/testing
"""
import json
import re
from string import punctuation

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
nltk.download('punkt')

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

# Loading the review data from json file 
data = []
with open('yelp_dataset/yelp_academic_dataset_review.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
del data
print('have dataframe')

df.drop(['review_id','user_id',	'business_id','useful','funny','cool', 'date'], axis=1, inplace=True)

df['stars'] = df['stars'].astype(int)

# ## Identifying and extracting target

# currently using:
# convert star rating to    negative, neutral,  positive
#                           [1,0,0]   [0,1,0]   [0,0,1]

df['negative'] = (df['stars'] < 3).astype(int)
df['neutral'] = (df['stars'] == 3).astype(int)
df['positive'] = (df['stars'] > 3).astype(int)

df.drop(['stars'], axis=1, inplace=True)

df['text'] = df['text'].apply(clean_text)

train, test = train_test_split(df, train_size=0.8, random_state=42)
print('train shape: ', train.shape)
print('test shape:', test.shape)

train[['negative', 'neutral', 'positive']].to_pickle('data/train_labels')
train[['text']].to_pickle('data/train_text')
print('train data saved')

test[['negative', 'neutral', 'positive']].to_pickle('data/test_labels')
test[['text']].to_pickle('data/test_text')
print('test data saved')