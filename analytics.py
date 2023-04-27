"""
Evaluating and analyzing the compiled model
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization

from load_dataset import clean_text

label_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}
labels = ['negative', 'neutral', 'positive']
model = keras.models.load_model('models/compiled_at_2023-04-27 002722.366698')

def predict_label(text, model, encoder):
    text = clean_text(text)
    text = encoder(np.array([text]))
    prediction = np.argmax(model(text))
    return label_dict[prediction]



def load_test_data():
    with open('data/test_encoded_text', 'rb') as f:
        X_test = np.load(f)
    Y_test = pd.read_pickle('data/test_labels')
    return X_test, Y_test

def make_encoder():
    with open('vocab', 'rb') as f:
        vocab = np.load(f)
    return TextVectorization(max_tokens=1000, output_sequence_length=250, vocabulary=vocab)

def examples():
    # for these simple examples
    example_review = 'I loved this restaurant, the food was great'
    encoder = make_encoder()
    prediction = predict_label(example_review, model, encoder, dirty=True, encoded=False)
    if prediction == 'positive':  # should be negative
        print('Prediction was correct!')
    else:
        print('incorrect!')
    example_review1 = 'food was yucky, service even worse'
    prediction = predict_label(example_review1, model, encoder, dirty=True, encoded=False)
    if prediction == 'negative':
        print('prediction correct!')
    else:
        print('incorrect!')
    example_review2 = "Eh it was okay, I don't have super strong feelings about it. Not the worst, not the best."
    prediction = predict_label(example_review2, model, encoder, dirty=True, encoded=False)
    if prediction == 'neutral':
        print('prediction correct!')
    else:
        print('incorrect!')


def load_test_data():
    with open('data/test_encoded_text', 'rb') as f:
        X_test = np.load(f)
    Y_test = pd.read_pickle('data/test_labels')
    return X_test, Y_test

def predict_test_data():
    X_test, _ = load_test_data()

    predictions = model.predict(X_test)
    predictions = np.argmax(predictions, axis=1)
    with open('predictions', 'wb') as f:
        np.save(f,predictions)

def create_conf_mat():
    with open('predictions', 'rb') as f:
        predictions = np.load(f)
    predictions = pd.Series(predictions).map(label_dict)
    _, y_test = load_test_data()
    y_test = np.argmax(y_test, axis=1)
    y_test = pd.Series(y_test).map(label_dict)
    # Create confusion matrix and plot it
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.savefig('confusion_matrix.png')

create_conf_mat()
