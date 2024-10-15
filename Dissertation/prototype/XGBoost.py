# Importing necessary libraries
import time
import xgboost as xgb
from dask_ml.model_selection import RandomizedSearchCV
from dask_ml.model_selection import GridSearchCV
import numpy as np

from models.basemodelclass import BaseModelClass
class XGBoost(BaseModelClass):
    def __init__(self):
        super().__init__(model_name = 'XGBoost')
        pass

    def _build_model(self):
        print('Not used method! Use train_model() instead this method.')
    
    def _build_best_model(self, best_params):
        self.best_model = xgb.XGBClassifier(
        **best_params,  # Apply best parameters
        verbosity=0  # Suppress unnecessary warnings and logs
        )
        return self.best_model
    
    def convert_to_dmatrix(self,data,y):
        d_matrix = xgb.DMatrix(data, label=y)
        return d_matrix
    
    def train_model(self, data, y):
        # Convert to DMatrix (XGBoost's data format)
        d_matrix = self.convert_to_dmatrix(data, y)

        # Define model parameters for XGBoost
        params = {
            'objective': 'binary:logistic',  # Binary classification
            'max_depth': 6,
            'eta': 0.3,
            'verbosity': 1,
            'eval_metric': 'logloss',
            'tree_method': 'gpu_hist'  # Use GPU for training
        }

        # Train the model using GPU
        start_time = time.time()
        self.model = xgb.train(params, d_matrix, num_boost_round=100)
        self.training_time = time.time() - start_time

        return self.model
    
    def train_best_model(self, data, y, best_params):
        # Train the model using GPU
        self.best_model = self._build_best_model(best_params)
        start_time = time.time()
        self.best_model.fit(data,y)
        self.best_training_time = time.time() - start_time

        return self.best_model
    
    def random_search(self, data, y, n_iter, cv, verbos, random_state, n_jobs):
        model = xgb.XGBClassifier(tree_method='gpu_hist', use_label_encoder=False, eval_metric='logloss', verbosity=verbos)

        param_dist = {
            'learning_rate': np.linspace(0.05, 0.2, 5),  
            'n_estimators': np.arange(50, 200, 250),  
            'max_depth': np.arange(3, 8, 1),  
            'min_child_weight': np.arange(1, 4, 1),
            'subsample': np.linspace(0.6, 0.9, 3),
            'colsample_bytree': np.linspace(0.6, 0.9, 3), 
            'gamma': np.linspace(0, 0.3, 3), 
            'tree_method': ['hist'],  
            'device': ['cuda'],
        }

        # RandomizedSearchCV (Dask)
        self.random_search_cv = RandomizedSearchCV(
            model, 
            param_distributions=param_dist, 
            n_iter=n_iter,  
            scoring='accuracy', 
            cv=cv,  
            random_state=random_state,
            n_jobs=n_jobs
        )

        start_time = time.time()
        self.random_search_cv.fit(data.toarray(), y)
        self.random_search_time = time.time() - start_time

        print(f"Best parameters found: {self.random_search_cv.best_params_}")

    def grid_search(self, data, y, cv, verbos, n_jobs, best_params=None):
        # Initialize the XGBoost classifier with base settings
        model = xgb.XGBClassifier(
            tree_method='gpu_hist',  # Use GPU for histogram optimization
            use_label_encoder=False,  # Disable label encoder warning
            eval_metric='logloss',  # Use log loss as the evaluation metric
            verbosity=verbos  # Suppress unnecessary warnings and logs
        )

        # Dynamically create the parameter grid based on the best parameters found by RandomizedSearchCV
        param_grid = {
            'learning_rate': np.linspace(best_params['learning_rate'] - 0.01, best_params['learning_rate'] + 0.01, 3),  
            'n_estimators': [best_params['n_estimators']],  
            'max_depth': [best_params['max_depth']],  
            'min_child_weight': [best_params['min_child_weight']],  
            'subsample': np.linspace(best_params['subsample'] - 0.00, best_params['subsample'] + 0.05, 3),  
            'colsample_bytree': np.linspace(best_params['colsample_bytree'] - 0.00, best_params['colsample_bytree'] + 0.05, 3),  
            'gamma': np.linspace(best_params['gamma'] - 0.00, best_params['gamma'] + 0.05, 3),  
            'tree_method': ['hist'],  # Use GPU
            'device': ['cuda'],  # GPU usage
        }
        # # Dynamically create the parameter grid based on the best parameters found by RandomizedSearchCV
        # param_grid = {
        #     'learning_rate': np.linspace(best_params['learning_rate'] - 0.01, best_params['learning_rate'] + 0.01, 3),  
        #     'n_estimators': [best_params['n_estimators'] - 50, best_params['n_estimators'], best_params['n_estimators'] + 50],  
        #     'max_depth': [best_params['max_depth'] - 1, best_params['max_depth'], best_params['max_depth'] + 1],  
        #     'min_child_weight': [best_params['min_child_weight'] - 1, best_params['min_child_weight']],  
        #     'subsample': np.linspace(best_params['subsample'] - 0.05, best_params['subsample'] + 0.05, 3),  
        #     'colsample_bytree': np.linspace(best_params['colsample_bytree'] - 0.05, best_params['colsample_bytree'] + 0.05, 3),  
        #     'gamma': np.linspace(best_params['gamma'] - 0.05, best_params['gamma'] + 0.05, 3),  
        #     'tree_method': ['hist'],  # Use GPU
        #     'device': ['cuda'],  # GPU usage
        # }

        # Proceed with GridSearchCV using this dynamically generated param_grid
        self.grid_search_cv = GridSearchCV(
            model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=cv,
            n_jobs=n_jobs,
        )

        # Fit the model
        start_time = time.time()
        self.grid_search_cv.fit(data.toarray(), y)
        self.grid_search_time = time.time() - start_time
        # Print the best parameters and model
        print(f"Best parameters found by GridSearchCV: {self.grid_search_cv.best_params_}")



