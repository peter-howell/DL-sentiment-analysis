import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization

from load_dataset import clean_text

label_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}

def predict_label(text, model, encoder, dirty=False, encoded=True):
    if dirty:
        text = clean_text(text)
    if not encoded:
        text = encoder(np.array([text]))
    return label_dict[np.argmax(model(text))]

model = keras.models.load_model('models/compiled_at_2023-04-27 002722.366698')

def load_encoded_text():
    with open('data/test_encoded_text', 'rb') as f:
        X_test = np.load(f)
    return X_test

# can use this to make 
def make_encoder():
    with open('vocab', 'rb') as f:
        vocab = np.load(f)
    return TextVectorization(max_tokens=1000, output_sequence_length=250, vocabulary=vocab)

# Y_test = pd.read_pickle('data/test_labels')
def examples():
    # for these simple examples
    example_review = 'I loved this restaurant, the food was great'
    encoder = make_encoder()
    prediction = predict_label(example_review, model, encoder, dirty=True, encoded=False)
    if prediction == 'positive':  # should be negative
        print('Prediction was correct!')
    else:
        print('incorrect!')
    example_review = 'food was yucky, service even worse'
    prediction = predict_label(example_review, model, encoder, dirty=True, encoded=False)
    if prediction == 'negative':
        print('prediction correct!')
    else:
        print('incorrect!')
    example_review = "Eh it was okay, I don't have super strong feelings about it. Not the worst, not the best."
    prediction = predict_label(example_review, model, encoder, dirty=True, encoded=False)
    if prediction == 'neutral':
        print('prediction correct!')
    else:
        print('incorrect!')
