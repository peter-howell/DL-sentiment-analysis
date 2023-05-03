"""Code for Task 1 of the DS 3010 final project
Loads business and review data
We used a bag-of-words approach to encode data from reviews for businesses, and then used a random forest to predict business attributes
"""

import json

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from analytics import plot_conf_mat
from load_datasets import load_review_dataset, load_business_dataset, clean_reviews



def words_from_series(series: pd.Series) -> list[str]:
    """gets the words contained in the text of a Series

    Args:
        series (pd.Series): series of text

    Returns:
        list[str]: list of all the words
    """
    return series.str.split(expand=True).stack().reset_index(drop=True).to_list()



def get_business_attributes(business_data: dict, n_attributes=None):
    """Gets the business attributes present in `business_data`, optionally getting only the top `n` boolean attributes

    Args:
        business_data (dict): dictionary with business_id as keys and attributes as values
        n_attributes (int, optional): number of boolean attributes to return. if None, then all are returned

    Returns:
        list[str]: list of the (top `n_attributes`) boolean business attributes
    """
    attributes = []
    for attrs in business_data.values():
        # add the business attributes that this business has
        attributes.extend(attrs)

    if n_attributes is None:
        return attributes
    # get the n business attributes had by most businesses
    top_attrs = pd.Series(attributes).value_counts().head(n_attributes).index
    return top_attrs.tolist()

def get_businesses(data_source: dict, attribute: str) -> set[str]:
    """Gets the businesses that have the given attribute

    Args:
        data_source (dict): collection of all the businesses and their attributes
        attribute (str): the target attribute
    
    Returns:
        set[str]: set of business_id's that have the given attribute
    """
    businesses = set()
    for bus_id in data_source:
        if attribute in data_source[bus_id]:
            businesses.add(bus_id)
    return businesses

def get_reviews(review_df: pd.DataFrame, businesses: set[str], min_reviews=50) -> pd.DataFrame:
    """Gets reviews for the given businesses only if the businesses have received more than `min_reviews` reviews

    Args:
        review_df (pd.Dataframe): dataframe of the reviews and business_id's
        businesses (set[str]): initial set of the business_id's to include
        min_reviews (int): the minimum number of reviews needed for a business to be included
    Returns:
        pd.DataFrame: desired rows of business id's and reviews
    """
    # Filter rows based on businesses
    filtered_df = review_df[review_df['business_id'].isin(businesses)]
    
    # Count the number of reviews for each business
    review_counts = filtered_df['business_id'].value_counts()
    
    # Get businesses with more than min_reviews
    valid_businesses = review_counts[review_counts >= min_reviews].index
    
    # Filter rows based on the valid businesses
    filtered_df = filtered_df[filtered_df['business_id'].isin(valid_businesses)]
    
    return filtered_df

class BoWClassifier:
    """
    A classifier using scikit-learn machine learning models and a bag-of-words approach to predict a business attribute
    """
    def __init__(self, clf, bag_limit=1500):
        """Create a bag-of-words classifier using a model from sklearn

        Args:
            clf : the model to use for the classification
            bag_limit (int): maximum number of words to have in the bag
        """
        self.vectorizer = CountVectorizer(max_features=bag_limit)
        self.clf = clf

    def _fill_bag(self, X: pd.DataFrame) -> dict:
        """fills bag of words for each business in the DataFrame

        Args:
            X (pd.DataFrame): dataframe of business IDs

        Returns:
            dict: dictionary with business ID as key and list of words as values
        """
        business_bags = {}

        # for each business, look at the reviews
        for business_id, df_i in X.groupby('business_id'):
            # get all the words in all the reviews for this business
            bus_words = words_from_series(df_i['text'])
            # store the words this business had in its bag
            business_bags[business_id] = bus_words
        return business_bags
    
    def fit(self, X: pd.DataFrame, y):
        """fits a model to the given review and business

        Args:
            X (DataFrame): dataframe of the reviews
            y : labels, if there are k businesses then y should be of length k
        """
        business_bags = self._fill_bag(X)

        # fit the vectorizer with the words found in the reviews
        X = self.vectorizer.fit_transform([' '.join([word for word in bag]) for bag in business_bags.values()])
        # fit the model
        self.clf.fit(X, y)

    def predict(self, X: pd.DataFrame):
        """predicts the label for a business attribute given the reviews

        Args:
            X (DataFrame): reviews for the business(es)

        Return: one of the class labels
        """
        # fill the bags of words 
        business_bags = self._fill_bag(X)
        # transform bags into sequence of tokens
        X = self.vectorizer.transform([' '.join([word for word in bag]) for bag in business_bags.values()])
        return self.clf.predict(X)

def load_and_clean():
    """
    Loads and processes the datasets and cleans the reviews before saving them to a file 
    because cleaning can take a while it is helpful to save after this step
    """
    
    
if __name__ == '__main__':
    fpath = "/Users/peterhowell/Downloads/yelp_dataset/yelp_academic_dataset_business.json"
    # fpath = "/Users/peterhowell/Downloads/yelp_dataset/business.json"
    # minumum number of reviews for a business needed to be included
    min_reviews = 50
    business_data = load_business_dataset(fpath)
    print('loaded business dataset')
    top_attributes = get_business_attributes(business_data, n_attributes=5)
    print('found the top 5 attributes to be: ' + ', '.join(top_attributes))
    review_df = load_review_dataset()
    # drop the columns that aren't business_id or text
    review_df = review_df[['business_id', 'text']]

    # remove reviews from businesses that have under 50 reviews and aren't in business_data
    review_df = get_reviews(review_df, set(business_data.keys()), min_reviews=min_reviews)
    # clean the reviews
    review_df = clean_reviews(review_df)
    # update business_data to remove businesses that were removed from review_df
    bus_ids = set(review_df['business_id'])

    business_data = {k: v for k, v in business_data.items() if k in bus_ids}
    
    # go through and make a classifier for each of the top attributes
    for attribute in top_attributes:
        print('current attribute: %s' %attribute)

        # get businesses that have this attribute
        bus_ids = get_businesses(business_data, attribute)
        # get reviews from those if they have more than 50 reviews
        reviews_i = get_reviews(review_df, bus_ids)
        # update businesses for those that have more than 50 reviews
        bus_ids = set(reviews_i['business_id'])

        print('number of businesses for attribute %s: %i' %(attribute, len(bus_ids)))
        # split businesses into training and testing
        train_bus, test_bus = train_test_split(list(bus_ids), test_size=0.2, random_state=42)
        # store labels for the business attributes
        train_labels = [business_data[business_id][attribute] for business_id in train_bus]
        test_labels  = np.vstack([business_data[business_id][attribute] for business_id in test_bus])

        # split reviews into training and testing
        train_reviews = reviews_i[reviews_i['business_id'].isin(train_bus)]
        test_reviews = reviews_i[reviews_i['business_id'].isin(test_bus)]

        # make the model with classifier of choice
        model = BoWClassifier(RandomForestClassifier())
        # fit model on current set of businesses
        model.fit(train_reviews, train_labels)
        print('model has been trained')

        predictions = model.predict((test_reviews))

        accuracy = accuracy_score(test_labels, predictions)
        print('test accuracy: %.4f \n\n' %(accuracy))
        plot_conf_mat(test_labels, predictions, attribute + 'conf_mat.png', labels=None)
    
