# User-Input-Classifier

Classifies the user input into three respective classes:
1. Q - QUESTION
2. S - STATEMENT
3. C - CHAT/SMALL TALK

Features extracted using self designed N - grams. Used a Random Forrest Classifier to train the model. 

FILES
-----
features.py - Extracts relavent features from the data
features_csv.py - Converts and formats the extracted features into a csv format file
MODEL_RUN.py - Trains the model on a Random ForrestClassifier. 

RUN
---
Run the model using RUN_MODEL.py

Input the string while running the python file, eg, python RUN_MODEL.py 'Is earth flat?'
