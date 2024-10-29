# Importing necessary libraries
# importing libraries
from keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop
from scikeras.wrappers import KerasClassifier
from functools import partial
import time
import pickle
import numpy as np

from basemodelclass import BaseModelClass
class BiLSTM(BaseModelClass):
    def __init__(self, tokenizer, epochs=15, batch_size=64, max_feature = 5000, max_length = 100, output_dim = 128, optimizer = 'adam', embedding_dim = 32, verbose=0):
        super().__init__(model_name = 'BiLSTM')
        self.max_feature = max_feature
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size=batch_size
        self.output_dim = output_dim
        self.optimizer = optimizer
        self.embedding_dim = embedding_dim
        self.verbose = verbose
        self.tokenizer=tokenizer
    
    def __create_bilstm_model(self, dropout=0, recurrent_dropout=0, optimizer='adam',output_dim=1, lstm_units=100, embedding_dim=64,learning_rate=0.001):
        model = Sequential()
        model.add(Embedding(input_dim=len(self.tokenizer.index_word)+1, output_dim=embedding_dim, input_length=self.max_length))
        model.add(Bidirectional(LSTM(units=lstm_units, dropout=dropout, recurrent_dropout=recurrent_dropout)))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(output_dim, activation="sigmoid"))
        model.build(input_shape=(None,self.max_length))
        
        if optimizer == 'adam':
            selected_optimizer = Adam()
        elif optimizer == 'rmsprop':
            selected_optimizer = RMSprop()

        model.compile(optimizer=selected_optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def _build_model(self,patience=3):
                # Partially apply the model parameters
        build_fn = partial(self.__create_bilstm_model,
                           dropout=0.01, recurrent_dropout=0.01, optimizer='adam',output_dim=1, lstm_units=100, embedding_dim=64,learning_rate=0.1)
        
        model = KerasClassifier(
            build_fn=build_fn, 
            dropout=0, 
            recurrent_dropout=0, 
            optimizer='adam',
            output_dim=1, 
            lstm_units=100, 
            embedding_dim=64,
            learning_rate=0.1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)])
        return model

    def _build_best_model(self, best_params , patience=3):
                # Partially apply the model parameters
        build_fn = partial(self.__create_bilstm_model,
                           dropout=0, recurrent_dropout=0, optimizer='adam',output_dim=1, lstm_units=100, embedding_dim=64,learning_rate=0.1)
        
        model = KerasClassifier(
            build_fn=build_fn, 
            dropout=best_params['dropout'], 
            recurrent_dropout=best_params['recurrent_dropout'], 
            optimizer=best_params['optimizer'],
            output_dim=1, 
            lstm_units=best_params['lstm_units'], 
            embedding_dim=best_params['embedding_dim'],
            callbacks=[EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)])
        return model
    
    def train_model(self, X_train_pad, y_train, X_test_pad, y_test,patience=3):
        self.model = self._build_model(patience=patience)

        start_time = time.time()
        self.model.fit(X_train_pad, y_train,batch_size=self.batch_size,epochs=self.epochs, validation_data=(X_test_pad, y_test))
        self.training_time = time.time() - start_time

    def train_best_model(self,X_train_pad, y_train, X_test_pad, y_test, best_params,patience=3):
        self.best_params = best_params
        self.best_model = self._build_best_model(patience=patience, best_params=best_params)

        start_time = time.time()
        self.best_model.fit(X_train_pad, y_train,batch_size=best_params['batch_size'],epochs=best_params['epochs']+5, validation_data=(X_test_pad, y_test))
        self.best_training_time = time.time() - start_time

    def random_search(self, X_train_pad, y_train, X_test_pad, y_test, n_iter=200, cv=3, random_state=42, n_jobs=1,patience=3):
        origin_param_dist = self.model.get_params()
        
        
        
        # Defining the parameter grid
        param_dist = {
            'lstm_units': np.unique([50, 100, origin_param_dist['lstm_units']]),            # Number of LSTM units
            'embedding_dim': np.unique([64, origin_param_dist['embedding_dim']]),          # Embedding dimensions
            'dropout': np.unique([0.01, 0.2, 0.4, origin_param_dist['dropout']]),         # Dropout rates
            'recurrent_dropout':np.unique([0.0, 0.01,  0.2, origin_param_dist['recurrent_dropout']]),
            'output_dim':[1],
            # 'learning_rate':np.unique([0.0001, 0.001, 0.01, origin_param_dist['learning_rate']]),
            'optimizer': np.unique(['adam', 'rmsprop']),
            'epochs': [self.epochs],
            'batch_size': [self.batch_size],
            # 'lstm_units': np.unique([50, 100, 150, origin_param_dist['lstm_units']]),            # Number of LSTM units
            # 'embedding_dim': np.unique([32, 64, 128,origin_param_dist['embedding_dim']]),          # Embedding dimensions
            # 'dropout': np.unique([0.0, 0.2, 0.4,origin_param_dist['dropout']]),         # Dropout rates
            # 'recurrent_dropout':np.unique([0.0, origin_param_dist['recurrent_dropout']]),
            # 'output_dim':[1],
            # 'learning_rate':np.unique([0.0001, 0.001, 0.01,origin_param_dist['learning_rate']]),
            # 'batch_size': np.unique(self.batch_size, self.batch_size*2),                    # Batch size
            # 'epochs': [self.epochs],                              # Number of epochs
            # 'optimizer': np.unique(['adam', 'rmsprop']) # Optimizers
        }

        model = self._build_model(patience=patience)
        # Running RandomizedSearchCV
        self.random_search_cv = RandomizedSearchCV(estimator=model,
                                        param_distributions=param_dist,
                                        n_iter=n_iter,  # Number of parameter settings to sample
                                        cv=cv,       # Number of folds for cross-validation
                                        verbose=self.verbose,
                                        n_jobs=n_jobs,
                                        random_state=random_state, error_score='raise')
        start_time = time.time()
        # Training with RandomizedSearchCV
        self.random_search_cv.fit(X_train_pad, y_train, validation_data=(X_test_pad, y_test))
        self.random_search_time = time.time() - start_time

    def grid_search(self, X_train_pad, y_train, X_test_pad, y_test, best_params, cv=3, n_jobs=1,patience=3):

        lstm_units = best_params['lstm_units'] if best_params['lstm_units'] is not None else 1
        embedding_dim = best_params['embedding_dim'] if best_params['embedding_dim'] is not None else 1
        dropout = best_params['dropout'] if best_params['dropout'] is not None else 1
        recurrent_dropout = best_params['recurrent_dropout'] if best_params['recurrent_dropout'] is not None else 1
        # learning_rate = best_params['learning_rate'] if best_params['learning_rate'] is not None else 1
        optimizer = best_params['optimizer'] if best_params['optimizer'] is not None else 'adam'

        param_grid = {
            # 'lstm_units': [best_params['lstm_units']],            # Number of LSTM units
            # 'embedding_dim': [best_params['embedding_dim']],          # Embedding dimensions
            # 'dropout': [best_params['dropout']],         # Dropout rates
            # 'recurrent_dropout':[best_params['recurrent_dropout']],
            # 'learning_rate':[best_params['learning_rate']],
            # 'batch_size': [int(best_params['batch_size']*0.9), best_params['batch_size'], int(best_params['batch_size']*1.1)],                    # Batch size
            # 'epochs': [best_params['epochs']+5],                              # Number of epochs
            # 'optimizer': ['adam', 'rmsprop'] # Optimizers

            'lstm_units': [lstm_units],            # Number of LSTM units
            'embedding_dim': [embedding_dim],          # Embedding dimensions
            'dropout': [dropout*0.9, dropout, dropout*1.1],         # Dropout rates
            'recurrent_dropout':[recurrent_dropout*0.9, recurrent_dropout, recurrent_dropout*1.1],
            # 'learning_rate':[learning_rate*0.9, learning_rate, learning_rate*1.1],
            'optimizer': [optimizer],
            'batch_size': [self.batch_size],
            'epochs': [self.epochs]
        }

        model = self._build_model(patience=patience)
        # Running GridSearchCV
        self.grid_search_cv = GridSearchCV(estimator=model,
                                param_grid=param_grid,
                                cv=cv,
                                verbose=self.verbose,
                                n_jobs=n_jobs)

        start_time= time.time()
        self.grid_search_cv.fit(X_train_pad, y_train, validation_data=(X_test_pad, y_test))
        self.grid_search_time = time.time() - start_time

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
