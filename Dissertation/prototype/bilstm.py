# Importing necessary libraries
# importing libraries
from keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.layers import Embedding

from transformers import BertTokenizer, TFBertForSequenceClassification, AdamWeightDecay
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras.models import Sequential
import tensorflow as tf
import time

# Load the TextPreprocessor class (assumed to be defined already)
from textpreprocessor import TextPreprocessor

class BiLSTM:
    def __init__(self, tokenizer, epochs=15, batch_size=64):
        self.model_name = 'BiLSTM'
        self.tokenizer = tokenizer
        self.epochs = epochs
        self.batch_size = batch_size
        pass
    
    def train_model(self, X_train_pad, y_train, X_test_pad, y_test):

        self.model = Sequential()
        self.model.add(Embedding(len(self.tokenizer.index_word)+1,64))
        self.model.add(Bidirectional(LSTM(100, dropout=0,recurrent_dropout=0)))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(1,activation="sigmoid"))

        self.model.compile("adam","binary_crossentropy",metrics=["accuracy"])
        self.model.summary()

        start_time = time.time()
        self.history = self.model.fit(X_train_pad, y_train,batch_size=self.batch_size,epochs=self.epochs, validation_data=(X_test_pad, y_test))
        self.training_time = time.time() - start_time

        return self.model


