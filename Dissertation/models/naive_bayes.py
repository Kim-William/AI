# Importing necessary libraries
from sklearn.naive_bayes import MultinomialNB
import time
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np

from basemodelclass import BaseModelClass
class Naive_Bayes(BaseModelClass):
    def __init__(self, verbose=1):
        super().__init__(model_name = 'Naive_Bayes')
        self.verbose = verbose

    def _build_model(self):
        print('Not used method! Use train_model() instead this method.')

    def _build_best_model(self, best_params):
        return MultinomialNB(alpha=best_params['alpha'], fit_prior=best_params['fit_prior'])

    def train_model(self,data,y):
        # Model building using Naive Bayes
        start_time = time.time()
        self.model = MultinomialNB()  # Naive Bayes classifier
        self.history = self.model.fit(data, y)
        self.training_time = time.time() - start_time

    def train_best_model(self,data,y, best_params):
        self.best_params = best_params
        self.best_model = self._build_best_model(best_params=best_params)
        start_time = time.time()
        self.best_model.fit(data, y)
        self.best_training_time = time.time()-start_time

    def random_search(self, data, y, n_iter, cv, random_state, n_jobs):
        origin_param_dist = self.model.get_params()
        # Define the initial parameter grid for RandomizedSearchCV
        param_dist = {
            # 'alpha': np.linspace(0.01, 1, 50),  # Smoothing parameter; commonly tuned in Naive Bayes
            # 'fit_prior': [True, False]  # Whether to learn class prior probabilities or not
            'alpha': np.append(np.linspace(0.001, 1, 2000), origin_param_dist['alpha']),  # Smoothing parameter; commonly tuned in Naive Bayes
            'fit_prior': [True, False]  # Whether to learn class prior probabilities or not
        }

        # Step 1: RandomizedSearchCV for rough tuning
        self.random_search_cv = RandomizedSearchCV(
            MultinomialNB(), 
            param_distributions=param_dist, 
            n_iter=n_iter,  # Number of random combinations to try
            scoring='accuracy', 
            cv=cv,  # fold cross-validation
            random_state=random_state,
            verbose=self.verbose,
            n_jobs=n_jobs  # Use all available CPU cores
        )

        start_time = time.time()
        # Run RandomizedSearchCV
        self.random_search_cv.fit(data, y)
        self.random_search_time = time.time() - start_time
        print(f"Best parameters from RandomizedSearchCV: {self.random_search_cv.best_params_}")

    def grid_search(self, data, y, best_params, cv, n_jobs):

        alpha = best_params['alpha'] if best_params['alpha'] is not None else 2
        param_grid = {
            'alpha': np.append(np.linspace(alpha - 1, alpha + 1.5, 10000), best_params['alpha']),  # Narrow range around best alpha
            'fit_prior': [best_params['fit_prior']]  # Use the best fit_prior value found
        }

        # GridSearchCV for fine-tuning
        self.grid_search_cv = GridSearchCV(
            MultinomialNB(), 
            param_grid=param_grid, 
            scoring='accuracy', 
            cv=cv,  # 5-fold cross-validation
            verbose=self.verbose,
            n_jobs=n_jobs  # Use all available CPU cores
        )

        start_time = time.time()
        # Run GridSearchCV
        self.grid_search_cv.fit(data, y)
        self.grid_search_time = time.time() - start_time
        print(f"Best parameters from GridSearchCV: {self.grid_search_cv.best_params_}")

    def save_model_and_params(self, model_filename, best_model_filename, params_filename):
        super().save_model_and_params(model_filename, best_model_filename, params_filename)
