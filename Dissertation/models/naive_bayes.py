# Importing necessary libraries
from sklearn.naive_bayes import MultinomialNB
import time

class Naive_Bayes:
    def __init__(self):
        self.model_name = 'Naive_Bayes'
        pass
    
    def train_model(self, X_tfidf, y):

        # Model building using Naive Bayes
        start_time = time.time()
        self.model = MultinomialNB()  # Naive Bayes classifier
        self.history = self.model.fit(X_tfidf, y)
        self.training_time = time.time() - start_time

        return self.model


