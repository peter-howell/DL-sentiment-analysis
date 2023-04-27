import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
import re
from string import punctuation

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))    
label_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}

# find a way to not need this here
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

def predict_label(text, model, encoder):
	text = clean_text(text)
	text = encoder(np.array([text]))
	return label_dict[np.argmax(model(text))]

model = keras.models.load_model('models/compiled_at_2023-04-27 002722.366698')

# with open('data/test_encoded_text', 'rb') as f:
#     X_test = np.load(f)

with open('vocab', 'rb') as f:
    vocab = np.load(f)

encoder = TextVectorization(max_tokens=1000, output_sequence_length=250, vocabulary=vocab)

# Y_test = pd.read_pickle('data/test_labels')

something ='I hated this restaurant. Food was terrible. They took forever.'
prediction = predict_label(something, model, encoder)
if prediction == 'negative': # should be negative
	print('Prediction was correct!')
