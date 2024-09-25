# Importing necessary libraries
from sklearn.linear_model import LogisticRegression
import time

class Logistic_Regression:
    def __init__(self):
        self.model_name = 'Logistic_Regression'
        pass
    
    def train_model(self, X_tfidf, y):
        # Model building
        # Using Logistic Regression for sentiment classification
        start_time = time.time()
        self.model = LogisticRegression()
        self.history = self.model.fit(X_tfidf, y)
        self.training_time = time.time() - start_time

        return self.model


