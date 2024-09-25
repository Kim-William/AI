# Importing necessary libraries
from transformers import BertTokenizer, TFBertForSequenceClassification, AdamWeightDecay
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
import time

class BERT:
    def __init__(self, max_feature, max_length, epochs=15, batch_size=64):
        self.model_name = 'BERT'
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        pass
    
    def train_model(self, X_train, y_train, X_test, y_test):
        # Initialize the BERT tokenizer
        self.tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')

        # Tokenizing the datasets
        self.X_train_tokens = self.tokenizer_bert(
            text=list(X_train),
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf',
            return_token_type_ids=False,
            return_attention_mask=True
        )

        self.X_test_tokens = self.tokenizer_bert(
            text=list(X_test),
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf',
            return_token_type_ids=False,
            return_attention_mask=True
        )

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
            {'input_ids': self.X_train_tokens['input_ids'], 'attention_mask': self.X_train_tokens['attention_mask']},
            y_train,
            validation_data=({'input_ids': self.X_test_tokens['input_ids'], 'attention_mask': self.X_test_tokens['attention_mask']}, y_test),
            epochs=self.epochs,
            batch_size=self.batch_size
        )

        self.training_time = time.time() - start_time

        return self.model


