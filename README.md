# Up_vote_Prediction

Predicting up votes for news articles based on the headline, creation time, author and news category

### Description of files/folders:

1) data_preparation.py 

Processes the dataset, creates train, test pickle files and saves them in the data directory

2) news_data.py

Contains the custom pytorch Dataset class

3) Data 

Contains the tokenized word vectors for training and testing in pickle format, and the mappings for word to index

4) Models

Contains the trained models

### Training from Scratch:

Step 1: Run the data_preparation.py file. This will create train, test pickle objects in the data directory.

Step 2: Run the train.py with the set hyperparameters

### Testing using pretrained model:

Run test.py using the model saved in models directory



