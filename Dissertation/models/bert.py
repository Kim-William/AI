# Importing necessary libraries
from transformers import BertTokenizer, TFBertForSequenceClassification, AdamWeightDecay, TFBertModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D

from tensorflow.keras.models import Model
from keras_tuner import HyperParameters, RandomSearch

import numpy as np
import time

from basemodelclass import BaseModelClass
class BERT(BaseModelClass):
    def __init__(self, max_length, epochs=15, batch_size=64, verbose=1):
        super().__init__(model_name='BERT')
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose=verbose
        self.tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')

    def _build_model(self):
        pass

    def _build_best_model(self, best_params):
        pass

    def __make_tokens(self, data):
        return self.tokenizer_bert(
            text=list(data),
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf',
            return_token_type_ids=False,
            return_attention_mask=True
        )
    
    def train_model(self, X_train, y_train, X_test, y_test):
        X_train_tokens = self.__make_tokens(X_train)
        X_test_tokens = self.__make_tokens(X_test)

        # Define BERT Model
        self.model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

        # Use Hugging Face's AdamWeightDecay optimizer
        optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
        # Compile the model using a standard loss function
        loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

        # Model Summary
        self.model.summary()

        # Train the model
        start_time = time.time()
        self.history = self.model.fit(
            {'input_ids': X_train_tokens['input_ids'], 'attention_mask': X_train_tokens['attention_mask']},
            y_train,
            validation_data=({'input_ids': X_test_tokens['input_ids'], 'attention_mask': X_test_tokens['attention_mask']}, y_test),
            epochs=self.epochs,
            batch_size=self.batch_size
        )

        self.training_time = time.time() - start_time

    def train_best_model(self,data,y, best_params):
        pass

    def random_search(self, X_train, y_train, X_test, y_test, max_trials = 20, executions_per_trial=1, n_jobs=1):
        def build_bert_model(hp):
            input_ids = Input(shape=(100,), dtype='int32', name="input_ids")
            attention_mask = Input(shape=(100,), dtype='int32', name="attention_mask")
            
            bert_model = TFBertModel.from_pretrained("bert-base-uncased")
            bert_output = bert_model(input_ids, attention_mask=attention_mask)[0] 
            pooled_output = GlobalAveragePooling1D()(bert_output)  
            
            dense = Dense(units=hp.Int("units", min_value=32, max_value=128, step=16), activation='relu')(pooled_output)
            output = Dense(2, activation='softmax')(dense)
            
            model = Model(inputs=[input_ids, attention_mask], outputs=output)
            
            learning_rate = hp.Choice("learning_rate", values=[1e-5, 2e-5, 3e-5])
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])
            
            return model
        X_train_tokens = self.__make_tokens(X_train)
        X_test_tokens = self.__make_tokens(X_test)

        self.random_search_cv = RandomSearch(
            build_bert_model,
            objective="val_accuracy",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory="bert_tuning",
            project_name="bert_sentiment_analysis",
            n_jobs = n_jobs
        )

        train_data = (
            {"input_ids": X_train_tokens["input_ids"], "attention_mask": X_train_tokens["attention_mask"]},
            y_train
        )
        val_data = (
            {"input_ids": X_test_tokens["input_ids"], "attention_mask": X_test_tokens["attention_mask"]},
            y_test
        )

        self.random_search_cv.search(
            x=train_data[0],
            y=train_data[1],
            validation_data=val_data,
            epochs=self.epochs,
            batch_size=self.batch_size
        )

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