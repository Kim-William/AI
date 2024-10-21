# Importing necessary libraries
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import pickle
import time

from basemodelclass import BaseModelClass
class Logistic_Regression(BaseModelClass):
    def __init__(self, verbose=1):
        super().__init__(model_name='Logistic_Regression')
        self.verbose=verbose
        pass
    
    def _build_model(self):
        model = LogisticRegression()
        return model

    def _build_best_model(self, best_params):
        model = LogisticRegression(**best_params)
        return model

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
    
    # def random_search(self, X_tfidf, y, n_iter=200, cv=3, random_state=42, n_jobs=-1):
    #     model = self._build_model()
    #     origin_param_dist = self.model.get_params()
    #     param_dist= {
    #         'C':np.append(np.logspace(-1,1,10), origin_param_dist['C']),#np.linspace(origin_param_dist['C']-0.5, origin_param_dist['C']+0.5, 100),
    #         # 'max_iter': np.unique(np.linspace(0, origin_param_dist['max_iter']*1.5, 10).astype(int)),
    #         'penalty': ['l1', 'l2', 'elasticnet'],
    #         # 'solver': ['liblinear', 'saga','lbfgs'],
    #         'max_iter':np.unique([100, 200, 500,origin_param_dist['max_iter']]),
    #         'solver': np.unique(['saga','lbfgs', origin_param_dist['solver']]),
    #         'tol': [1e-4, 1e-3, 1e-2],
    #         'class_weight': [None, 'balanced'],
    #         'l1_ratio': [0.1, 0.5, 0.9]  # elasticnet 사용 시 적용
    #     }

    #     self.random_search_cv = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv, verbose=self.verbose, random_state=random_state, n_jobs=n_jobs)

    #     start_time = time.time()
    #     self.random_search_cv.fit(X_tfidf, y)
    #     self.random_search_time = time.time() - start_time

    def random_search_elasticnet(self, X_tfidf, y, n_iter=200, cv=3, random_state=42, n_jobs=-1):
        model = self._build_model()
        origin_param_dist = self.model.get_params()
        param_dist= {
            'C':np.append(np.logspace(-1,1,10), origin_param_dist['C']),#np.linspace(origin_param_dist['C']-0.5, origin_param_dist['C']+0.5, 100),
            # 'max_iter': np.unique(np.linspace(0, origin_param_dist['max_iter']*1.5, 10).astype(int)),
            'penalty': ['elasticnet'],
            'max_iter':np.unique([100, 200, 500,origin_param_dist['max_iter']]),
            'solver': np.unique(['saga','lbfgs', origin_param_dist['solver']]),
            'tol': [1e-4, 1e-3, 1e-2, origin_param_dist['tol']],
            'class_weight': [None, 'balanced',origin_param_dist['class_weight']],
            'l1_ratio': [0.1, 0.5, 0.9,origin_param_dist['l1_ratio']]  # elasticnet 사용 시 적용
        }

        self.random_search_cv_elasticnet = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv, verbose=self.verbose, random_state=random_state, n_jobs=n_jobs)

        start_time = time.time()
        self.random_search_cv_elasticnet.fit(X_tfidf, y)
        self.random_search_time_elasticnet = time.time() - start_time

    def random_search(self, X_tfidf, y, n_iter=200, cv=3, random_state=42, n_jobs=-1):
        model = self._build_model()
        origin_param_dist = self.model.get_params()
        param_dist= {
            'C':np.append(np.logspace(-1,2,50), origin_param_dist['C']),#np.linspace(origin_param_dist['C']-0.5, origin_param_dist['C']+0.5, 100),
            # 'max_iter': np.unique(np.linspace(0, origin_param_dist['max_iter']*1.5, 10).astype(int)),
            'penalty': ['l1', 'l2'],
            # 'solver': ['liblinear', 'saga','lbfgs'],
            'class_weight': [origin_param_dist['class_weight']],
            'max_iter':np.unique([100, 200, 500,origin_param_dist['max_iter']]),
            'solver': np.unique(['saga','lbfgs', origin_param_dist['solver']]),
            'tol': np.unique([1e-4, 1e-3, 1e-2, origin_param_dist['tol']]),
        }

        self.random_search_cv = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv, verbose=self.verbose, random_state=random_state, n_jobs=n_jobs)

        start_time = time.time()
        self.random_search_cv.fit(X_tfidf, y)
        self.random_search_time = time.time() - start_time

    def grid_search(self, X_tfidf, y, best_params, cv=3, n_jobs=-1):
        model = self._build_model()

        
        if best_params['penalty']=='elasticnet':
            param_dist= {
            'C':np.append(np.linspace(best_params['C']-0.25, best_params['C']+0.25, 10), best_params['C']),
            # 'max_iter': [best_params['max_iter']], #np.unique(np.linspace(int(best_params['max_iter']*0.75), best_params['max_iter']*1.25, 20).astype(int)),
            'penalty': [best_params['penalty']],
            'max_iter':[best_params['max_iter']],
            'class_weight': [best_params['class_weight']],
            'solver': [best_params['solver']],
            'tol': np.append(np.linspace(best_params['tol']*0.75, best_params['tol']*1.25, 10), best_params['tol']),
            'l1_ratio': [best_params['l1_ratio']] 
        }
        else:
            param_dist= {
            'C':np.append(np.linspace(best_params['C']-0.25, best_params['C']+0.25, 10), best_params['C']),
            # 'max_iter': [best_params['max_iter']], #np.unique(np.linspace(int(best_params['max_iter']*0.75), best_params['max_iter']*1.25, 20).astype(int)),
            'penalty': [best_params['penalty']],
            'max_iter':[best_params['max_iter']],
            'class_weight': [best_params['class_weight']],
            'solver': [best_params['solver']],
            'tol': np.append(np.linspace(best_params['tol']*0.75, best_params['tol']*1.25, 10), best_params['tol'])
        }
            
        self.grid_search_cv = GridSearchCV(model, param_grid=param_dist, cv=cv, verbose=self.verbose, n_jobs=n_jobs)
        
        start_time = time.time()
        self.grid_search_cv.fit(X_tfidf, y)
        self.grid_search_time = time.time() - start_time

    def save_model_and_params(self, model_filename, best_model_filename, params_filename):
        super().save_model_and_params(model_filename, best_model_filename, params_filename)