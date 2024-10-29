# Importing necessary libraries
import time
import xgboost as xgb
from dask_ml.model_selection import RandomizedSearchCV
from dask_ml.model_selection import GridSearchCV
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
import cupy as cp

from basemodelclass import BaseModelClass
class XGBoost(BaseModelClass):
    def __init__(self, verbose=1):
        super().__init__(model_name = 'XGBoost')
        self.verbose = verbose
        pass

    def _build_model(self):
        print('Not used method! Use train_model() instead this method.')
    
    def _build_best_model(self, best_params):
        self.best_model = xgb.XGBClassifier(
        **best_params,  # Apply best parameters
        verbosity=self.verbose  # Suppress unnecessary warnings and logs
        )
        return self.best_model
    
    def convert_to_dmatrix(self,data,y):
        d_matrix = xgb.DMatrix(data, label=y)
        return d_matrix
    
    def _load_data_on_gpu(self, data, y):
        
        # Convert features to dense array if it's sparse
        if hasattr(data, 'toarray'):
            dense_data = data.toarray()
        else:
            dense_data = data.values  # Assuming it's a DataFrame

        # Ensure the data is float32
        dense_data = dense_data.astype(np.float32)
        # Convert to CuPy array
        gpu_data = cp.asarray(dense_data)
        # Convert target variable to CuPy array
        y_gpu = cp.asarray(y.astype(np.float32))
        
        return gpu_data, y_gpu
    
    def train_model(self, data, y):
        # Convert to DMatrix (XGBoost's data format)
        basic_params={
            'colsample_bytree': 0.5499999999999999,
            'device': 'cuda',
            'eval_metric': 'logloss',
            'gamma': 0.15,
            'learning_rate': 0.02,
            'min_child_weight': 1,
            'n_estimators': 100,
            'subsample': 0.11,
            'tree_method': 'hist',
            'use_label_encoder': False
            }
        self.model = self._build_best_model(basic_params)
        # Train the model using GPU
        start_time = time.time()
        self.model.fit(data,y)
        # self.model = xgb.train(params, d_matrix, num_boost_round=100)
        self.training_time = time.time() - start_time
    
    def train_best_model(self, data, y, best_params):
        self.best_params = best_params
        # Train the model using GPU
        self.best_model = self._build_best_model(best_params)
        start_time = time.time()
        self.best_model.fit(data,y)
        # self.best_model.fit(data,y)
        self.best_training_time = time.time() - start_time
    
    def random_search(self, data, y, n_iter, cv, random_state, n_jobs):
        model = xgb.XGBClassifier(tree_method='gpu_hist', use_label_encoder=False, eval_metric='logloss', verbosity=self.verbose)
        origin_param_dist = model.get_params()
        param_dist = {
            'learning_rate': np.append(np.linspace(0.05, 0.2, 5), origin_param_dist['learning_rate']),  
            'n_estimators': np.append(np.arange(50, 200, 250), origin_param_dist['n_estimators']),  
            'max_depth': np.append(np.arange(3, 8, 1), origin_param_dist['max_depth']),  
            'min_child_weight': np.append(np.arange(1, 4, 1), origin_param_dist['min_child_weight']),
            'subsample': np.append(np.linspace(0.6, 0.9, 3), origin_param_dist['subsample']),
            'colsample_bytree': np.append(np.linspace(0.6, 0.9, 3), origin_param_dist['colsample_bytree']), 
            'gamma': np.append(np.linspace(0, 0.3, 3), origin_param_dist['gamma']), 
            'tree_method': ['hist'],  
            'device': ['cuda'],
            'eval_metric':['logloss'],
            'use_label_encoder':[False]
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
        self.random_search_cv.fit(data,y)
        self.random_search_time = time.time() - start_time

        print(f"Best parameters found: {self.random_search_cv.best_params_}")

    def _append_to_param_dist(self, param_dist, param_name, param_value, min, max, count):
        if param_value is not None:
            param_dist[param_name]= np.unique(np.append(np.linspace(min, max, count), param_value))
        return param_dist

    def grid_search(self, data, y,best_params, cv, n_jobs):
        # Initialize the XGBoost classifier with base settings
        model = xgb.XGBClassifier(
            tree_method='gpu_hist',  # Use GPU for histogram optimization
            use_label_encoder=False,  # Disable label encoder warning
            eval_metric='logloss',  # Use log loss as the evaluation metric
            verbosity=self.verbose  # Suppress unnecessary warnings and logs
        )

        learning_rate = best_params['learning_rate'] #if best_params['learning_rate'] is not None else 0.01
        n_estimators = best_params['n_estimators']# if best_params['n_estimators'] is not None else 50
        min_child_weight = best_params['min_child_weight'] #if best_params['min_child_weight'] is not None else 2
        subsample = best_params['subsample'] #if best_params['subsample'] is not None else 0.06
        colsample_bytree = best_params['colsample_bytree'] #if best_params['colsample_bytree'] is not None else 0.6
        gamma = best_params['gamma'] #if best_params['gamma'] is not None else 0.15
        max_depth = best_params['max_depth']
        # param_grid = self._append_to_param_dist(param_grid, 'learning_rate', best_params['learning_rate'], learning_rate-0.01, learning_rate+0.01, 3)
        # param_grid = self._append_to_param_dist(param_grid, 'n_estimators', best_params['n_estimators'], n_estimators-50, n_estimators+50, 3)
        # param_grid = self._append_to_param_dist(param_grid, 'min_child_weight', best_params['min_child_weight'], min_child_weight-1, min_child_weight+1, 3)
        # param_grid = self._append_to_param_dist(param_grid, 'subsample', best_params['subsample'], subsample-0.05, subsample+0.05, 3)
        # param_grid = self._append_to_param_dist(param_grid, 'colsample_bytree', best_params['colsample_bytree'], colsample_bytree-0.05, colsample_bytree+0.05, 3)
        # param_grid = self._append_to_param_dist(param_grid, 'gamma', best_params['gamma'], gamma-0.05, gamma+0.05, 3)
        # print(param_grid)
        # # Dynamically create the parameter grid based on the best parameters found by RandomizedSearchCV
        param_grid = {
            'learning_rate': [learning_rate],  
            'n_estimators': [n_estimators],  
            'min_child_weight': [min_child_weight - 1, min_child_weight, min_child_weight+1],  
            'subsample': [subsample],  
            'colsample_bytree': [colsample_bytree],  
            'gamma': np.unique(np.append(np.linspace(gamma - 0.05, gamma + 0.05, 3), gamma)),  
            'tree_method': ['hist'],  # Use GPU
            'device': ['cuda'],
            'eval_metric':['logloss'],
            'use_label_encoder':[False],
            'max_depth':[max_depth-1, max_depth, max_depth+1]
        }

        # Proceed with GridSearchCV using this dynamically generated param_grid
        self.grid_search_cv = GridSearchCV(
            model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=cv,
            n_jobs=n_jobs
        )

        # Fit the model
        start_time = time.time()
        self.grid_search_cv.fit(data,y)
        self.grid_search_time = time.time() - start_time
        # Print the best parameters and model
        print(f"Best parameters found by GridSearchCV: {self.grid_search_cv.best_params_}")

    def save_model_and_params(self, model_filename, best_model_filename, params_filename):
        super().save_model_and_params(model_filename, best_model_filename, params_filename)