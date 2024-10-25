
## Overview
This project implements hyperparameter tuning for various models using Logistic_Regression, XGBoost, Naive_bayse, RNN, CNN, and BiLSTM architectures. Follow the steps below to set up and run the models.

## Installation Instructions

### 1. Install Requirements

To install the necessary packages, open your terminal and navigate to the project directory. Then, run the following command:

[bash]

pip install -r requirements.txt

This command will install all the dependencies listed in the requirements.txt file.

### 2. Running the Models
After installing the requirements, you can run the models using the following commands:


#### For the traditional models such as Logistic_regression, XGBoost, Naive_bayse, execute:
[bash]

python hyper_traditional.py


#### For the RNN model, execute:
[bash]

python hyper_rnn.py


#### For the CNN model, execute:
[bash]

python hyper_cnn.py


#### For the BiLSTM model, execute:
[bash]

python hyper_bilstm.py

Make sure to choose the model you wish to run by using the appropriate command.

### 3. Output and Results

After running the models, the evaluation results will be saved in an Excel file located in the Output/result directory. 
This file will contain the following information:
Training time, Accuracy

Comparison of:
Basic model, RandomizedSearchCV results, GridSearchCV results, Final best model

## Conclusion
Follow these instructions to successfully run the models and view the results. If you have any questions or issues, please contact the author.

rodzl12382@gmail.com