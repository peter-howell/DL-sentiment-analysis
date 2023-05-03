# DS3010-Final
Final project for DS3010: Computational Data Intelligence. Our big focus was on our task 2, creating a neural network for sentiment analysis. We hope that this repository is easily accessible and can be easily adapted to the needs of others.

## Task 2 Information:

The model achieved 88.59% accuracy on test data. The compiled, trained model is stored in the models/ directory, and the vocabulary binary is also here in vocab. 

To create the model we ran load_dataset.py, then encode_text.py and then model.py.

load_dataset.py loads the yelp data from the review file and stores the relevant features in temporary files, extracts and cleans the text of reviews as well as the review ratings, and splits the data 80/20 into training/testing. The file also provides some utility that is used in task 1.

encode_text.py uses text vectorization to encode the reviews as sequences of word indices and stores the vocabulary found from the training data

model.py that creates, trains, saves, and performs minor evaluation on the neural network for sentiment analysis

analytics.py loads the compiled model and generates the confusion matrix. It also allows for examples in plain text to be encoded, embedded, and procesed by the model. The confusion matrix code is used in task 1 as well.

The remaining python files (except for task_1.py) generate the wordclouds, tables, and histograms related to the word frequency.

## Task 1 Information

Due to bias in the dataset, our approach yielded models that almost exclusively predicted positives because most of the businesses that had attributes had them because they were positive.

Task 1 used some code from load_dataset and analytics, but for the most part is contained in task_1.py.

We looked for the top 5 most common boolean attributes and trained a random forest classifier using a bag-of-words approach that can be applied to many different classification algorithms. 
