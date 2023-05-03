"""
Code for Task 2 that uses text vectorization to encode the reviews as sequences of word indices
and stores the vocabulary found from the training data 
"""

import numpy as np
import pandas as pd
from tensorflow.keras.layers import TextVectorization

def load_text(fname:str) -> pd.Series:
    """loads text data from a file

    Args:
        fname (str): name of the file
    Returns:
        (pd.Series): text stored in the file
    """
    return pd.read_pickle(fname)['text'].astype(str)

def encode_text(encoder: TextVectorization, text: pd.Series, fname:str):
    """Encodes text data into vectorized format and store in a file

    Args:
        encoder (TextVectorization): adapted encoder to use
        text (pd.Series): text to encode
        fname (str): where to store the encoded text
    """
    encoded_text = np.array(encoder(text))
    with open(fname, 'wb') as f:
        np.save(f, encoded_text)

# load the cleaned training text to adapt the encoder and create the vocabulary
reviews = load_text('data/train_text')

max_tokens = 1000 # max size of vocab
sequence_len = 250 # max length of review

# adapt the encoder with the training data
encoder = TextVectorization(max_tokens=max_tokens, output_sequence_length=sequence_len)
encoder.adapt(reviews)

print('encoder adapted')

# saving the encoder's vocabulary so it can be used for adapting later if needed
vocab = np.array(encoder.get_vocabulary())
with open('vocab', 'wb') as f:
    np.save(f, vocab)
print('vocabulary saved')

# save the encoded training data
encode_text(encoder, reviews, 'data/train_encoded_text')
print('train data encoded')

reviews = load_text('data/test_text')
# encode and save the test data with the adapted encoder
encode_text(encoder, reviews, 'data/test_encoded_text')
print('test data encoded')
