# DS3010-Final
Final project for DS3010: Computational Data Intelligence. 

## Task 2 Information:

The model achieved 88.59% accuracy on test data. The compiled, trained model is stored in the models/ directory, and the vocabulary binary is also here in vocab. 

To create the model we ran load_dataset.py, then encode_text.py and then model.py.

load_dataset.py loads the yelp data from the review file and stores the relevant features in temporary files, extracts and cleans the text of reviews as well as the review ratings, and splits the data 80/20 into training/testing

encode_text.py uses text vectorization to encode the reviews as sequences of word indices and stores the vocabulary found from the training data

model.py that creates, trains, saves, and performs minor evaluation on the neural network for sentiment analysis

analytics.py loads the compiled model and generates the confusion matrix. It also allows for examples in plain text to be encoded, embedded, and procesed by the model.

The remaining python files (except for task_1.py) generate the wordclouds, tables, and histograms related to the word frequency.

## Task 1 Information

Task 1 used some task 2 code from load_dataset and analytics, but for the most part is contained in task_1.py. Something else more information
