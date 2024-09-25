# Importing necessary libraries
import time
from keras.layers import Dense
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential

class CNN:
    def __init__(self, max_feature, max_length, epochs=15, batch_size=64):
        self.model_name = 'CNN'
        self.max_feature = max_feature
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        pass
    
    def train_model(self, X_train_pad, y_train,X_test_pad, y_test):
        # CNN Model Construction
        embedding_dim = 128  # Embedding layer dimension

        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.max_feature, output_dim=embedding_dim, input_length=self.max_length))
        self.model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))  # Binary classification (0 or 1)

        # Compile the model
        self.model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        start_time = time.time()
        self.history= self.model.fit(X_train_pad, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_test_pad, y_test))
        self.training_time = time.time() - start_time

        return self.model


