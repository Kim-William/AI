# Importing necessary libraries
import time 
from functools import partial
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, MaxPooling1D, Flatten, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
import pickle

from models.basemodelclass import BaseModelClass
class CNN(BaseModelClass):
    def __init__(self, max_feature = 5000, max_length = 100, epochs=15, batch_size=64, output_dim = 128, optimizer = 'adam', embedding_dim = 32, verbose=0):
        super().__init__(model_name = 'CNN')
        self.max_feature = max_feature
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size=batch_size
        self.output_dim = output_dim
        self.optimizer = optimizer
        self.embedding_dim = embedding_dim
        self.verbose = verbose
    
    def _create_cnn_model(self, filters=128, kernel_size=5, pool_size=2, dropout_rate=0.25, learning_rate=0.001, optimizer='adam'):
        model = Sequential()
        model.add(Embedding(input_dim=self.max_feature, output_dim=self.embedding_dim, input_length=self.max_length))
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=pool_size))  # MaxPooling1D 사용
        model.add(Dropout(dropout_rate))
        model.add(Flatten())  # Flatten 추가
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification (0 or 1)

        model.build(input_shape=(None, self.max_length))
        if optimizer == 'adam':
            selected_optimizer = Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            selected_optimizer = RMSprop(learning_rate=learning_rate)
        model.compile(optimizer=selected_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def _build_model(self, patience= 3):
        # Partially apply the model parameters
        build_fn = partial(self._create_cnn_model,
                           filters=128, kernel_size=5, pool_size=2, dropout_rate=0.25, learning_rate=0.001, optimizer='adam')
        
        model = KerasClassifier(
            build_fn=build_fn, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            verbose=self.verbose, 
            kernel_size=5, 
            filters=128, 
            pool_size=2, 
            dropout_rate=0.25, 
            learning_rate = 0.001,
            callbacks=[EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)])

        return model
    
    def _build_best_model(self, best_params, patience= 3):
        build_fn = partial(self._create_cnn_model,
                           filters=128, kernel_size=5, pool_size=2, dropout_rate=0.25, learning_rate=0.001, optimizer='adam')
        searched_model = KerasClassifier(
            build_fn=build_fn,
            filters=best_params['filters'], 
            kernel_size=best_params['kernel_size'], 
            pool_size=best_params['pool_size'], 
            dropout_rate=best_params['dropout_rate'], 
            learning_rate = best_params['learning_rate'],
            batch_size=best_params['batch_size'],
            epochs=best_params['epochs'],
            optimizer=best_params['optimizer'],
            verbose=self.verbose,
            callbacks=[EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
        )
        return searched_model

    def train_model(self, data, y, validation_data):
        self.model = self._build_model()
        start_time = time.time()
        # Train the model
        if validation_data == None:
            self.history = self.model.fit(data, y, epochs=self.epochs, batch_size=self.batch_size, callbacks=self.early_stopping)
        else:
            self.history = self.model.fit(data, y, epochs=self.epochs, batch_size=self.batch_size, validation_data=validation_data, callbacks=self.early_stopping)
        self.training_time = time.time() - start_time

    def train_best_model(self,data,y, validation_data, best_params, patience= 3):
        self.best_params = best_params
        self.best_model = self._build_best_model(best_params = best_params, patience = patience)

        start_time = time.time()
        if validation_data == None:
            self.best_history = self.best_model.fit(data, y)
        else:
            self.best_history = self.best_model.fit(data, y, validation_data=validation_data)
        self.best_training_time = time.time() - start_time

    def random_search(self, data, y, validation_data=None, n_iter = 200, cv=3, random_state=42, n_jobs=-1, patience=3):
        model = self._build_model(patience=patience)
        param_dist = {
            'filters': [128, 256],
            'kernel_size': [3, 5],
            'pool_size': [2, 3],               
            'dropout_rate': [0.25, 0.5, 0.75],    
            'learning_rate': [0.0005, 0.001],  
            'batch_size': [64],    
            'epochs': [3],
            'optimizer': ['adam', 'rmsprop'],               
        }

        self.random_search_cv = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=n_iter, cv=cv, verbose=self.verbose, n_jobs=n_jobs, random_state=random_state)
        start_time=time.time()
        
        if validation_data == None:
            self.random_search_cv.fit(data, y)
        else:
            self.random_search_cv.fit(data, y, validation_data=validation_data)
        self.random_search_time = time.time() - start_time

        print(f"Best parameters found: {self.random_search_cv.best_params_}")

    def grid_search(self, data, y, validation_data, best_params, cv=3, n_jobs=-1, patience=3):
        model = self._build_model(patience=patience)
        param_dist =  {
            'filters': [best_params['filters']],
            'kernel_size': [best_params['kernel_size']],
            'pool_size': [best_params['pool_size']],               
            'dropout_rate': [best_params['dropout_rate']],    
            'learning_rate': [best_params['learning_rate'], best_params['learning_rate']*1.1],  
            'batch_size': [best_params['batch_size']],    
            'epochs': [best_params['epochs']],
            'optimizer': [best_params['optimizer']],     

            # 'filters': [int(best_params['filters']*0.9), best_params['filters'], int(best_params['filters']*1.1)],
            # 'kernel_size': [best_params['kernel_size']-1, best_params['kernel_size'], best_params['kernel_size']+1],
            # 'pool_size': [best_params['pool_size'] -1, best_params['pool_size'], best_params['pool_size']+1],               
            # 'dropout_rate': [best_params['dropout_rate']*0.9,best_params['dropout_rate'],best_params['dropout_rate']*1.1],    
            # 'learning_rate': [best_params['learning_rate']*0.9, best_params['learning_rate'], best_params['learning_rate']*1.1],  
            # 'batch_size': [int(best_params['batch_size']*0.8), best_params['batch_size'], int(best_params['batch_size']*1.2)],    
            # 'epochs': [int(best_params['epochs']*0.8), best_params['epochs'], int(best_params['epochs']*1.2)],
            # 'optimizer': [best_params['optimizer']],               
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