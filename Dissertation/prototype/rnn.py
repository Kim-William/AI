# Importing necessary libraries
import time 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

class RNN:
    def __init__(self, max_feature, max_length, epochs=15, batch_size=64):
        self.model_name = 'RNN'
        self.max_feature = max_feature
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        pass
    
    def train_model(self, X_train_pad, y_train,X_test_pad, y_test):
        # RNN Model Building
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.max_feature, output_dim=128, input_length=self.max_length))
        self.model.add(SimpleRNN(64))  # Using a Simple RNN layer
        self.model.add(Dense(1, activation='sigmoid'))  # Binary classification

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        start_time = time.time()
        # Train the model
        self.history = self.model.fit(X_train_pad, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_test_pad, y_test))
        self.training_time = time.time() - start_time

        return self.model


