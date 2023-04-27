"""
Code that creates, trains, evaluates, and saves the neural network for sentiment analysis
"""
import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
import datetime


with open('data/train_encoded_text', 'rb') as f:
    X_train = np.load(f)

Y_train = pd.read_pickle('data/train_labels')

# Defining the architecture of the model
model = Sequential([
    # Embedding layer - converts sequence of word indices into a sequence of vectors, each vector encodes a word
    Embedding(input_dim=1000, output_dim=50, mask_zero=True),
    # LSTM layer: retains important information in the sequence but forgets unimportant
    LSTM(units=64, dropout=0.2, recurrent_dropout=0.2),
    # Output layer: 3 units for three categories: negative, neutral, positive
    Dense(units=3, activation='softmax')
])

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=5, batch_size=32)

print('model compiled successfully')

with open('data/test_encoded_text', 'rb') as f:
    X_test = np.load(f)

Y_test = pd.read_pickle('data/test_labels')

loss, accuracy = model.evaluate(X_test, Y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

model_name = 'compiled_at_' + str(datetime.datetime.now())
model.save('models/' + model_name)