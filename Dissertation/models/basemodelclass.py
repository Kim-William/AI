from abc import ABC, abstractmethod
import pickle

from tensorflow.keras.callbacks import EarlyStopping

class BaseModelClass(ABC):
    def __init__(self, model_name:str):
        self.model_name = model_name

        self.model = None
        self.history = None
        self.training_time = None

        self.best_model=None
        self.best_history = None
        self.best_training_time=None
        self.best_params=None

        self.random_search_cv=None
        self.random_search_time =None

        self.grid_search_cv=None
        self.grid_search_time =None
        self.early_stopping = EarlyStopping(
            monitor='val_loss',  # Monitor the validation loss
            patience=2,  # Number of epochs with no improvement after which training will be stopped
            restore_best_weights=True  # Restore the weights of the best epoch
        )
        
    def _build_model(self):
        pass
    def _build_best_model(self, best_params):
        pass
    def train_model(self,data,y):
        pass
    def train_best_model(self,data,y, best_params):
        pass
    def random_search(self, data, y, n_iter, cv, verbos, random_state, n_jobs):
        pass
    def grid_search(self, data, y, best_params, n_iter, cv, verbos, random_state, n_jobs):
        pass
    def save_model_and_params(self, model_filename, best_model_filename, params_filename):
        # Save the initial trained model
        with open(model_filename, 'wb') as model_file:
            pickle.dump(self.model, model_file)
        print(f"Model saved to {model_filename}")

        # Save the best model found by hyperparameter tuning
        with open(best_model_filename, 'wb') as best_model_file:
            pickle.dump(self.best_model, best_model_file)
        print(f"Best model saved to {best_model_filename}")

        # Save the best hyperparameters
        with open(params_filename, 'wb') as params_file:
            pickle.dump(self.best_params, params_file)
        print(f"Best parameters saved to {params_filename}")