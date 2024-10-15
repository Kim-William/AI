# Importing necessary libraries
import time
import xgboost as xgb

class XGBoost:
    def __init__(self):
        self.model_name = 'XGBoost'
        pass
    
    def train_model(self, X_tfidf, y):
        # Convert to DMatrix (XGBoost's data format)
        d_matrix = self.convert_to_dmatrix(X_tfidf, y)

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
    
    def convert_to_dmatrix(self,X_tfidf,y):
        d_matrix = xgb.DMatrix(X_tfidf, label=y)
        return d_matrix


