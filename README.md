# User-Input-Classifier
-----------------------
It classifies the user input in the model into the following three classes:
1. Question - Q
2. Statement - S
3. Chat/Small talk - C

Evaluation
----------
Evaluated the model on Random Forests, Support Vector Machines and K Nearest Neighbors Classifier to determine the best choice of classifier to be used.

The Precision, Recall anf F-1 score metrics are:

1. Random Forest
Precision - 0.766835016835
Recall - 0.756756756757
F-1 - 0.757319966062


2. K Nearest Neighbors
Precision - 0.726943346509
Recall - 0.702702702703
F-1 - 0.715927750411


3. Support Vector Machines
Precision - 0.681081918934
Recall - 0.675675675676
F-1 - 0.667170473205


Requirements
------------
Python3
NumPy
Sklearn
Pandas

How to Run
----------
Run the MODEL_RUN.py file as follows with the desired user input:

python3 MODEL_RUN.py "Is the earth flat"
