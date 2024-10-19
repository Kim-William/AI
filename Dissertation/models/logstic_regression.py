# Importing necessary libraries
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import pickle
import time

from models.basemodelclass import BaseModelClass
class Logistic_Regression(BaseModelClass):
    def __init__(self):
        super().__init__(model_name='Logistic_Regression')
        pass
    
    def _build_model(self):
        model = LogisticRegression()
        return model

    def _build_best_model(self, best_params):
        self.best_model = LogisticRegression(**best_params)
        return self.best_model

    def train_model(self, X_tfidf, y):
        # Model building
        # Using Logistic Regression for sentiment classification
        start_time = time.time()
        self.model = self._build_model()
        self.history = self.model.fit(X_tfidf, y)
        self.training_time = time.time() - start_time
    
    def train_best_model(self, X_tfidf, y, best_params):
        # Model building
        # Using Logistic Regression for sentiment classification
        self.best_params = best_params
        self.best_model = self._build_best_model(best_params)
        start_time = time.time()
        self.best_history = self.best_model.fit(X_tfidf, y)
        self.best_training_time = time.time() - start_time
    
    def random_search(self, X_tfidf, y, n_iter=200, cv=3, verbos=1, random_state=42, n_jobs=-1):
        model = self._build_model()

        param_dist= {
            'C':np.logspace(-5,3,200),
            'solver':['liblinear', 'saga','lbfgs'],
            # 'solver':['liblinear', 'saga','lbfgs'],
            # 'solver':['liblinear', 'saga','lbfgs'],
            # 'solver':['liblinear', 'saga','lbfgs'],
            'max_iter':[100, 200, 300],
            'penalty':['l1', 'l2']
        }

        self.random_search_cv = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv, verbose=verbos, random_state=random_state, n_jobs=n_jobs)

        start_time = time.time()
        self.random_search_cv.fit(X_tfidf, y)
        self.random_search_time = time.time() - start_time

    def grid_search(self, X_tfidf, y, best_params, n_iter=200, cv=3, verbos=1, random_state=42, n_jobs=-1):
        model = self._build_model()
        param_dist= {
                'C':np.logspace(best_params['C']-1,best_params['C']+1,1000),
                'solver':[best_params['solver']],
                # 'solver':['liblinear', 'saga','lbfgs'],
                # 'solver':['liblinear', 'saga','lbfgs'],
                # 'solver':['liblinear', 'saga','lbfgs'],
                'max_iter':[best_params['max_iter']-50, best_params['max_iter'], best_params['max_iter']+50],
                'penalty':[best_params['penalty']]
            }
            
        self.grid_search_cv = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv, verbose=verbos, random_state=random_state, n_jobs=n_jobs)
        
        start_time = time.time()
        self.grid_search_cv.fit(X_tfidf, y)
        self.grid_search_time = time.time() - start_time

    def save_model_and_params(self, model_filename, best_model_filename, params_filename):
        super().save_model_and_params(model_filename, best_model_filename, params_filename)