from abc import ABC, abstractmethod

class BaseModelClass(ABC):
    def __init__(self, model_name:str):
        self.model_name = model_name

        self.model = None
        self.history = None
        self.training_time = None

        self.best_model=None
        self.best_history = None
        self.best_training_time=None

        self.random_search_cv=None
        self.random_search_time =None

        self.grid_search_cv=None
        self.grid_search_time =None

        self.test_params=None
        
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
    def grid_search(self, data, y, n_iter, cv, verbos, random_state, n_jobs):
        pass