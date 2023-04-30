"""
Evaluates and analyzes the compiled model, plots a confusion matrix. Also allows for examples to be run
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

def predict_label(text: str, encoder: TextVectorization) -> str:
    """Wrapper for predicting the label of certain text using the compiled model and an encoder
    
    Args:
        text (str): text to predict
        encoder (TextVectorization): encoder to encode the text with, converts string to sequence of token indices
    Returns:
        (str): the predicted label of the text. Either positive, neutral, or negative
    """
    text = clean_text(text)
    text = encoder(np.array([text]))
    prediction = np.argmax(model(text))
    return label_dict[prediction]
    
def make_encoder() -> TextVectorization:
    """Makes a TextVectorization encoder using the vocabulary from the training dataset
    
    Returns:
        (TextVectorization): a text encoder adapted on the vocabulary obtained from the training dataset
    """
    with open('vocab', 'rb') as f:
        vocab = np.load(f)
    return TextVectorization(max_tokens=1000, output_sequence_length=250, vocabulary=vocab)

def example(text: str, label=None, encoder=None):
    """Evaluates an example. If given a known label, determines if the model's prediction was correct.

    Args:
        text (str): text to evaluate
        label (str): the true label for the example
        encoder (TextVectorization):  a text encoder adapted on the vocabulary obtained from the training dataset
    """
    if encoder is None:
        encoder = make_encoder()
    print('Review: \n\t' + text)
    prediction = predict_label(text, encoder)
    print('Prediction: \n\t' + prediction)
    if label is None:
        return
    if prediction == label:
        print('Correct!')
    else: 
        print('Incorrect. Expected ' + label)

def examples():
    """Runs a set of examples on the model
    """
    encoder = make_encoder()
    # for these simple examples
    example_review = 'I loved this restaurant, the food was great'
    example(example_review, 'positive', encoder)
    example_review = 'food was yucky, service even worse'
    example(example_review, 'negative', encoder)
    example_review = "Eh it was okay, I don't have super strong feelings about it. Not the worst, not the best."
    example(example_review, 'neutral', encoder)

def load_test_data():
    """Loads the encoded test data
    """
    with open('data/test_encoded_text', 'rb') as f:
        X_test = np.load(f)
    Y_test = pd.read_pickle('data/test_labels')
    return X_test, Y_test

def predict_test_data():
    """Preidcts labels for the test data and saves them
    """
    X_test, y_test = load_test_data()
    # convert y from length three vector to one integer, 
    y_test = y_test.idxmax(axis=1)

    predictions = model.predict(X_test)
    predictions = np.argmax(predictions, axis=1)
    with open('predictions', 'wb') as f:
        np.save(f,predictions)

def create_conf_mat():
    """Creates and saves plot of a confusion matrix of the model's predictions of the test data
    """
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
