# %%
import sys
sys.path.append('models')

# from google.colab import drive
# drive.mount('/content/drive')
# sys.path.append('/content/drive/MyDrive/Dissertation/models')
# sys.path.append('/content/drive/MyDrive/Dissertation')

# %%
# !pip install dask_ml
# !pip install scikeras
# import nltk
# nltk.download('stopwords')
# import nltk
# nltk.download('punkt')

# %%
# importing libraries
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from collections import defaultdict
import matplotlib.pyplot as plt

from logstic_regression import Logistic_Regression
from XGBoost import XGBoost
from naive_bayes import Naive_Bayes
from rnn import RNN
from cnn import CNN
# from models.bert import BERT
from bilstm import BiLSTM

# Load the TextPreprocessor class (assumed to be defined already)
from textpreprocessor import TextPreprocessor
from evaluation_visualization import Evaluation_Visualization

import warnings
warnings.filterwarnings("ignore")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# %%
NUM_SAMPLE = 10000
TEST_RATIO=0.2
BATCH_SIZE=512
EPOCHS = 10
MAX_WORD_COUNT = 5000
MAX_LENGTH = 222
OUTPUT_RESULT_DIR = "Output/result"
OUTPUT_MODELS_DIR = "Output/models"
USE_TEST_DATA = True
INPUT_DIR = f"InputData/sample_{NUM_SAMPLE}"

os.makedirs(OUTPUT_RESULT_DIR, exist_ok=True)
os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)

eval_and_visual = Evaluation_Visualization(out_result_dir= OUTPUT_RESULT_DIR, out_models_dir=OUTPUT_MODELS_DIR)

# %% [markdown]
# 00. Text Pre-Processing

# %%
# Initialize the Text Preprocessor
processor = TextPreprocessor(MAX_WORD_COUNT, MAX_LENGTH)

if USE_TEST_DATA:
    # Load data
    df_train = pd.read_csv(os.path.join(INPUT_DIR, 'train_cleaned.csv'))
    df_test = pd.read_csv(os.path.join(INPUT_DIR, 'test_cleaned.csv'))
    # df_test = processor.load_data()
    X_train = df_train['review']
    X_test = df_test['review']
    y_train = df_train['polarity']
    y_test = df_test['polarity']
    # X_train_seq_padded = pickle.load(os.path.join(INPUT_DIR, 'X_train_pad.pkl'))
    # X_test_seq_padded = pickle.load(os.path.join(INPUT_DIR, 'X_test_pad.pkl'))
else:
    # Load data
    df_train, df_test = processor.parallel_load_data()

    df_train_step1 = processor.remove_stopwords(df_train.copy())
    df_test_step1 = processor.remove_stopwords(df_test.copy())

    print('----------TRAIN DATA----------')
    df_train_step2 = processor.filter_by_length_of_sentence(df_train_step1.copy(),50)
    print('----------TEST DATA----------')
    df_test_step2 = processor.filter_by_length_of_sentence(df_test_step1.copy(),50)

    df_train_step3 = processor.sampling_data(df_train_step2, NUM_SAMPLE)
    df_test_step3 = processor.sampling_data(df_test_step2, int(NUM_SAMPLE*TEST_RATIO))

    # Preprocess data
    df_train_step3 = processor.map_polarity(df_train_step3.copy())
    df_test_step3 = processor.map_polarity(df_test_step3.copy())

    # Split data
    X_train, y_train = processor.split_data(df_train_step3)
    X_test, y_test = processor.split_data(df_test_step3)
    
    os.makedirs(INPUT_DIR, exist_ok=True)
    # Save data
    df_train_step3.to_csv(os.path.join(INPUT_DIR, 'train_cleaned.csv'), index=False)
    df_test_step3.to_csv(os.path.join(INPUT_DIR, 'test_cleaned.csv'), index=False)

X_train_tfidf, X_test_tfidf = processor.vectorize_text(X_train, X_test)
X_train_pad, X_test_pad = processor.tokenization_and_padding(X_train, X_test)

MAX_LENGTH = processor.max_length

# %% [markdown]
# 01. Logistic Regression

# %%
# 1. Train Model
logistic_regression = Logistic_Regression(verbose=1)
logistic_regression.train_model(X_train_tfidf, y_train)

# 2. Random SearchCV
logistic_regression.random_search(X_train_tfidf, y_train, n_iter=1500, cv=10, random_state=42, n_jobs=-1)
logistic_regression.random_search_elasticnet(X_train_tfidf, y_train, n_iter=1500, cv=10, random_state=42, n_jobs=-1)

_, best_params = eval_and_visual.compare_models_accuracy_and_get_best_params({'Original': logistic_regression.model,
                                                              'RandomizedSearchCV': logistic_regression.random_search_cv,
                                                              'ElasticNet': logistic_regression.random_search_cv_elasticnet}, X_test_tfidf, y_test)

# 3. Grid SearchCV
logistic_regression.grid_search(X_train_tfidf, y_train, cv=10, n_jobs=-1,best_params=best_params)

# 4. Train Best Model
logistic_regression.train_best_model(X_train_tfidf, y_train, logistic_regression.grid_search_cv.best_params_)

# 5. Evaluate and Save Models
eval_and_visual.evaluate_model_class(logistic_regression,X_test_tfidf, y_test)

logistic_regression.save_model_and_params(
    os.path.join(OUTPUT_MODELS_DIR, 'logistic_model.pkl'),
    os.path.join(OUTPUT_MODELS_DIR, 'logistic_best_model.pkl'),
    os.path.join(OUTPUT_MODELS_DIR, 'logistic_best_params.pkl')
    )

# %% [markdown]
# 02. XGBoost

# %%
# 1. Train Model
xgboost = XGBoost(verbose=1)
xgboost.train_model(X_train_tfidf.toarray(), y_train)

X_train_tfidf = X_train_tfidf.astype(np.float32)

# 2. Random SearchCV
xgboost.random_search(X_train_tfidf.toarray(), y_train, n_iter=2000, cv=10, random_state=42, n_jobs=-1)

_, best_params = eval_and_visual.compare_models_accuracy_and_get_best_params({'Original': xgboost.model,
                                                              'RandomizedSearchCV': xgboost.random_search_cv}, X_test_tfidf.toarray(), y_test)

# 3. Grid SearchCV
xgboost.grid_search(X_train_tfidf.toarray(), y_train, cv=10,  n_jobs=-1, best_params=best_params)

# 4. Train Best Model
xgboost.train_best_model(X_train_tfidf.toarray(), y_train, xgboost.grid_search_cv.best_params_)

# 5. Evaluate and Save Models
eval_and_visual.evaluate_model_class(xgboost, X_test_tfidf.toarray(), y_test)

xgboost.save_model_and_params(
    os.path.join(OUTPUT_MODELS_DIR, 'xgboost_model.pkl'),
    os.path.join(OUTPUT_MODELS_DIR, 'xgboost_best_model.pkl'),
    os.path.join(OUTPUT_MODELS_DIR, 'xgboost_best_params.pkl')
    )

# %% [markdown]
# 03. Naive Bayes

# %%
# 1. Train Model
naive_bayes = Naive_Bayes(verbose=1)
naive_bayes.train_model(X_train_tfidf, y_train)

# 2. Random SearchCV
# naive_bayes.random_search(X_train_tfidf, y_train, n_iter=30, cv=2, verbos=0, random_state=42, n_jobs=-1)
naive_bayes.random_search(X_train_tfidf, y_train, n_iter=5000, cv=10, random_state=42, n_jobs=-1)

_, best_params = eval_and_visual.compare_models_accuracy_and_get_best_params({'Original': naive_bayes.model,
                                                              'RandomizedSearchCV': naive_bayes.random_search_cv}, X_test_tfidf, y_test)

# 3. Grid SearchCV
# naive_bayes.grid_search(X_train_tfidf, y_train, naive_bayes.random_search_cv.best_params_, cv=2, verbos=1, n_jobs=-1)
naive_bayes.grid_search(X_train_tfidf, y_train, best_params=best_params, cv=10, n_jobs=-1)

# 4. Train Best Model
naive_bayes.train_best_model(X_train_tfidf, y_train, naive_bayes.grid_search_cv.best_params_)

# 5. Evaluate and Save Models
eval_and_visual.evaluate_model_class(naive_bayes,X_test_tfidf, y_test)

naive_bayes.save_model_and_params(
    os.path.join(OUTPUT_MODELS_DIR, 'naivebayes_model.pkl'),
    os.path.join(OUTPUT_MODELS_DIR, 'naivebayes_best_model.pkl'),
    os.path.join(OUTPUT_MODELS_DIR, 'naivebayes_best_params.pkl')
    )

# %% [markdown]
# 04. Recurrent Neural Network

# %%
# 1. Train Model
rnn = RNN(max_feature=5000, max_length=processor.max_length, epochs=EPOCHS, batch_size=BATCH_SIZE, output_dim=128, optimizer='adam', embedding_dim=32, rnn_unit=64, verbose=1)
rnn.train_model(X_train_pad, y_train, validation_data=(X_test_pad, y_test))

# 2. Random SearchCV
rnn.random_search(X_train_pad, y_train, (X_test_pad, y_test), n_iter=1000, cv=5,  random_state=42, n_jobs=1)

_, best_params = eval_and_visual.compare_models_accuracy_and_get_best_params({'Original': rnn.model,
                                                              'RandomizedSearchCV': rnn.random_search_cv}, X_test_pad, y_test)

# 3. Grid SearchCV
rnn.grid_search(X_train_pad, y_train, (X_test_pad, y_test), cv=5, n_jobs=1, best_params=best_params)

# 4. Train Best Model
rnn.train_best_model(X_train_pad,y_train, validation_data=(X_test_pad, y_test), best_params= rnn.grid_search_cv.best_params_)

# 5. Evaluate and Save Models
eval_and_visual.evaluate_model_class(rnn,X_test_pad, y_test)

rnn.save_model_and_params(
    os.path.join(OUTPUT_MODELS_DIR, 'rnn_model.pkl'),
    os.path.join(OUTPUT_MODELS_DIR, 'rnn_best_model.pkl'),
    os.path.join(OUTPUT_MODELS_DIR, 'rnn_best_params.pkl')
    )

# %%
# Plot for hist
eval_and_visual.plot_training_history_from_dict(rnn.model.history_, title="Initial Model Training History")

# Plot the training history from defaultdict data
eval_and_visual.plot_training_history_from_dict(rnn.best_model.history_, title="Best Model Training History")

# %% [markdown]
# 05. Convolutional Neural Network

# %%
# 1. Train Model
cnn = CNN(max_feature=processor.max_features, max_length=processor.max_length, epochs=EPOCHS, batch_size=BATCH_SIZE, output_dim=128, optimizer='adam', embedding_dim=32, verbose=1)
cnn.train_model(X_train_pad, y_train, (X_test_pad, y_test))

# 2. Random SearchCV
cnn.random_search(X_train_pad, y_train, (X_test_pad, y_test), n_iter=2, cv=2, random_state=42, n_jobs=1, patience=3)

_, best_params = eval_and_visual.compare_models_accuracy_and_get_best_params({'Original': cnn.model,
                                                              'RandomizedSearchCV': cnn.random_search_cv}, X_test_pad, y_test)

# 3. Grid SearchCV
cnn.grid_search(X_train_pad, y_train, (X_test_pad, y_test), best_params=best_params, cv=2, n_jobs=1, patience=3)

# 4. Train Best Model
cnn.train_best_model(X_train_pad, y_train, (X_test_pad, y_test), best_params=cnn.grid_search_cv.best_params_)

# 5. Evaluate and Save Models
eval_and_visual.evaluate_model_class(cnn,X_test_pad, y_test)

cnn.save_model_and_params(
    os.path.join(OUTPUT_MODELS_DIR, 'cnn_model.pkl'),
    os.path.join(OUTPUT_MODELS_DIR, 'cnn_best_model.pkl'),
    os.path.join(OUTPUT_MODELS_DIR, 'cnn_best_params.pkl')
    )

# %%
# Plot for hist
eval_and_visual.plot_training_history_from_dict(cnn.model.history_, title="Initial Model Training History")

# Plot the training history from defaultdict data
eval_and_visual.plot_training_history_from_dict(cnn.best_model.history_, title="Best Model Training History")

# %% [markdown]
# 06. Bidirectional Encoder Representations from Transformers(BERT)

# %%
# bert = BERT(max_length=processor.max_features, epochs=1, batch_size=BATCH_SIZE, verbose=1)
# bert.train_model(X_train, y_train, X_test, y_test)

# bert.random_search(X_train, y_train, X_test, y_test, max_trials=2, executions_per_trial=1, n_jobs=1)

# # Predict on test data
# y_pred_prob = bert_model.predict({'input_ids': bert.X_test_tokens['input_ids'], 'attention_mask': bert.X_test_tokens['attention_mask']}).logits
# y_pred_bert = np.argmax(y_pred_prob, axis=-1)

# %% [markdown]
# 07. Bidirectional Long Short-Term Memory (BiLSTM)

# %%
# 1. Train Model
bilstm = BiLSTM(processor.tokenizer, EPOCHS, BATCH_SIZE, verbose=1)
bilstm.train_model(X_train_pad, y_train, X_test_pad,y_test)

# 2. Random SearchCV
bilstm.random_search(X_train_pad, y_train, X_test_pad,y_test, n_iter=2, cv=2, random_state=42, n_jobs=1)

_, best_params = eval_and_visual.compare_models_accuracy_and_get_best_params({'Original': bilstm.model,
                                                              'RandomizedSearchCV': bilstm.random_search_cv}, X_test_pad, y_test)

# 3. Grid SearchCV
bilstm.grid_search(X_train_pad, y_train, X_test_pad, y_test, best_params= best_params, cv=2, n_jobs=1, patience=3)

# 4. Train Best Model
bilstm.train_best_model(X_train_pad, y_train, X_test_pad, y_test, best_params=bilstm.grid_search_cv.best_params_, patience=3)

# 5. Evaluate and Save Models
eval_and_visual.evaluate_model_class(bilstm,X_test_pad, y_test)

bilstm.save_model_and_params(
    os.path.join(OUTPUT_MODELS_DIR, 'bilstm_model.pkl'),
    os.path.join(OUTPUT_MODELS_DIR, 'bilstm_best_model.pkl'),
    os.path.join(OUTPUT_MODELS_DIR, 'bilstm_best_params.pkl')
    )

# %%
# Plot for hist
eval_and_visual.plot_training_history_from_dict(bilstm.model.history_, title="Initial Model Training History")

# Plot the training history from defaultdict data
eval_and_visual.plot_training_history_from_dict(bilstm.best_model.history_, title="Best Model Training History")

# %%
eval_and_visual.df_results


