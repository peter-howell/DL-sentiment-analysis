# DS3010-Final
Final project for DS3010: Computational Data Intelligence

To create the model, first load_dataset.py was run, then encode_text.py and then model.py.

The model achieved 88.59% accuracy on test data. The compiled, trained model is stored in the models/ directory, and the vocabulary binary is also here in vocab. 

analytics.py loads the compiled model and generates the confusion matrix. It also allows for examples in plain text to be encoded, embedded, and procesed by the model.

The remaining files generated the wordclouds, tables, and histograms related to the word frequency.
