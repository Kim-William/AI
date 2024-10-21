# Importing necessary libraries
import time 
from functools import partial
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping
import pickle

from basemodelclass import BaseModelClass
class RNN(BaseModelClass):
    def __init__(self, max_feature = 5000, max_length = 100, epochs=15, batch_size=64, output_dim = 128, optimizer = 'adam', embedding_dim = 32, rnn_unit=64, verbose=0):
        super().__init__(model_name = 'RNN')
        self.max_feature = max_feature
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size=batch_size
        self.output_dim = output_dim
        self.optimizer = optimizer
        self.embedding_dim = embedding_dim
        self.rnn_unit = rnn_unit
        self.verbose = verbose
    
    def _create_rnn_model(self, input_dim, embedding_dim, rnn_units, input_length, optimizer):
        # Define a simple RNN model
        model = Sequential()
        model.add(Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=input_length))
        model.add(SimpleRNN(rnn_units))
        model.add(Dense(1, activation='sigmoid'))
        model.build(input_shape=(None, input_length))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _build_model(self, patience = 3):
        # Partially apply the model parameters
        build_fn = partial(self._create_rnn_model,
                           input_dim=self.max_feature,
                           embedding_dim=self.embedding_dim,
                           input_length=self.max_length,
                           optimizer=self.optimizer)

        # Use the partially applied build function with KerasClassifier
        model = KerasClassifier(
            build_fn=build_fn, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            verbose=self.verbose, 
            rnn_units=self.rnn_unit,  # Set default rnn_units
            callbacks=[EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)],
            embedding_dim=self.embedding_dim
        )

        return model
    
    def _build_best_model(self, best_params, patience= 3):
        build_fn = partial(self._create_rnn_model,
                           input_dim=self.max_feature,
                           embedding_dim=self.embedding_dim,
                           input_length=self.max_length,
                           optimizer=self.optimizer)

        searched_model = KerasClassifier(
            build_fn=build_fn,
            input_dim=self.max_feature,
            input_length=self.max_length,
            rnn_units=best_params['rnn_units'],       
            embedding_dim=best_params['embedding_dim'],
            optimizer=best_params['optimizer'],
            epochs=best_params['epochs'] +10,
            batch_size=best_params['batch_size'],
            verbose=0,
            callbacks=[EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
        )
        return searched_model

    def train_model(self, data, y, validation_data):
        self.model = self._build_model()
        start_time = time.time()
        # Train the model
        if validation_data == None:
            self.model.fit(data, y, epochs=self.epochs, batch_size=self.batch_size, callbacks=self.early_stopping)
        else:
            self.model.fit(data, y, epochs=self.epochs, batch_size=self.batch_size, validation_data=validation_data, callbacks=self.early_stopping)
        self.training_time = time.time() - start_time

    def train_best_model(self,data,y, validation_data, best_params, patience= 3):
        self.best_params = best_params
        self.best_model = self._build_best_model(best_params = best_params, patience = patience)

        start_time = time.time()
        if validation_data == None:
            self.best_model.fit(data, y)
        else:
            self.best_model.fit(data, y, validation_data=validation_data)
        self.best_training_time = time.time() - start_time

    def random_search(self, data, y, validation_data=None, n_iter=200, cv=3, random_state=42, n_jobs=-1, patience=3):
        model = self._build_model(patience=patience)
        origin_param_dist = self.model.get_params()

        param_dist = {
            'rnn_units': [16, 32, 64, origin_param_dist['rnn_units']],
            'embedding_dim': [32, 64, 128, origin_param_dist['embedding_dim']],
            'optimizer': ['adam', 'rmsprop', origin_param_dist['optimizer']],
            'epochs': [origin_param_dist['epochs']-3, origin_param_dist['epochs'], origin_param_dist['epochs']+3, origin_param_dist['epochs']+10],
            'batch_size': [16, 32, 64, 128, origin_param_dist['epochs']]
        }

        self.random_search_cv = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            verbose=self.verbose,
            n_jobs=n_jobs,
            random_state=random_state
        )

        start_time = time.time()
        if validation_data is None:
            self.random_search_cv.fit(data, y)
        else:
            self.random_search_cv.fit(data, y, validation_data=validation_data)
        self.random_search_time = time.time() - start_time

        print(f"Best parameters found: {self.random_search_cv.best_params_}")

    def grid_search(self, data, y, validation_data, best_params, cv=3, n_jobs=-1, patience=3):
        model = self._build_model(patience=patience)

        rnn_units = best_params['rnn_units'] if best_params['rnn_units'] is not None else 1
        embedding_dim = best_params['embedding_dim'] if best_params['embedding_dim'] is not None else 1

        param_dist = {
            # 'rnn_units': [int(best_params['rnn_units']*0.9), best_params['rnn_units'], int(best_params['rnn_units']*1.1)],     
            # 'embedding_dim': [int(best_params['embedding_dim']*0.9),best_params['embedding_dim'], int(best_params['embedding_dim']*1.1)],      
            # 'optimizer': [best_params['optimizer']], 
            # 'epochs': [best_params['epochs']],                
            # 'batch_size': [best_params['batch_size']]   
            'rnn_units': [int(rnn_units*0.9),int(rnn_units*0.95), rnn_units, int(rnn_units*1.05), int(rnn_units*1.1)],     
            'embedding_dim': [int(embedding_dim*0.9),int(embedding_dim*0.95),embedding_dim, int(embedding_dim*1.05), int(embedding_dim*1.1)],      
            'optimizer': [best_params['optimizer']], 
            'epochs': [best_params['epochs']],                
            'batch_size': [best_params['batch_size']]          
        }

        self.grid_search_cv = GridSearchCV(
            estimator=model,
            param_grid=param_dist,
            cv=cv,       
            verbose=self.verbose,
            n_jobs=n_jobs
        )

        start_time = time.time()
        if validation_data == None:
            self.grid_search_cv.fit(data, y)
        else:
            self.grid_search_cv.fit(data, y, validation_data=validation_data)
        self.grid_search_time = time.time() - start_time
        print(f"Best parameters found by GridSearchCV: {self.grid_search_cv.best_params_}")
    
    def save_model_and_params(self, model_filename, best_model_filename, params_filename):
        # Save the initial trained model
        with open(model_filename, 'wb') as model_file:
            pickle.dump(self.model, model_file)
        print(f"Model saved to {model_filename}")

        # Save the best model found by hyperparameter tuning
        with open(best_model_filename, 'wb') as best_model_file:
            pickle.dump(self.best_model.model, best_model_file)
        print(f"Best model saved to {best_model_filename}")

        # Save the best hyperparameters
        with open(params_filename, 'wb') as params_file:
            pickle.dump(self.best_params, params_file)
        print(f"Best parameters saved to {params_filename}")