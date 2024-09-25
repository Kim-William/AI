
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
import numpy as np
import multiprocessing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import dask.dataframe as dd

class TextPreprocessor:
    # Example usage:
    PATH_TRAIN_DATA = 'InputData/train.csv'
    PATH_TEST_DATA = 'InputData/test.csv'
    DATA_SET_HEADER = ['polarity', 'title', 'review']
    stop_words = set(stopwords.words('english'))
    def __init__(self, max_features=5000, max_length = 100):
        """
        Constructor to initialize paths, column headers, and TF-IDF settings.
        
        :param max_features: [Maximum number of words in the vocabulary] or [Maximum number of words to use for TF-IDF vectorization]
        :param max_length: Maximum length of the input sequences
        """
        self.max_features = max_features
        self.max_length=max_length
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)

    def filter_by_length_of_sentence(self, X_train, percent=70):
        X_train['review_length'] = X_train['review'].apply(len)
        
        length_counts = X_train['review_length'].value_counts().reset_index()
        length_counts.columns = ['review_length', 'count']
        length_counts = length_counts.sort_values('review_length')
        
        length_counts['cumulative_sum'] = length_counts['count'].cumsum()
        length_counts['cumulative_percentage'] = 100 * length_counts['cumulative_sum'] / length_counts['count'].sum()
        
        threshold = length_counts[length_counts['cumulative_percentage'] >= percent].iloc[0]['review_length']
        self.max_length = threshold+1
        filtered_data = X_train[X_train['review_length'] <= threshold]
        
        return filtered_data
    
    def limit_length_of_sentence(self, X_train, max_length):
        """
        Limit the length of sentences in the 'review' column to the specified max_length.
        Any sentence exceeding the max_length will be truncated.

        Parameters:
        X_train: pd.DataFrame
            DataFrame containing the text data. It must have a 'review' column with text to be processed.
        max_length: int
            The maximum allowed length of a sentence. Sentences longer than this value will be truncated.

        Returns:
        pd.DataFrame
            A DataFrame with sentences truncated to the specified max_length.
        """

        # Truncate each review in 'review' column to max_length
        X_train['review_l'] = X_train['review'].apply(lambda x: ' '.join(x.split()[:max_length]) if len(x.split()) > max_length else x) # Use .copy() to avoid unintended effects on other columns
    
        
        return X_train


    def sampling_data(self,df, num_sample=10000):
        """
        Random sampling of {num_sample} reviews
        
        :param df: Input DataFrame
        :param num_sample: Number to sample
        :return: Sampled DataFrame
        """
        df = df.sample(n=num_sample, random_state=42)  # random_state ensures reproducibility

        return df

    def load_data(self, num_sample=0, test_ratio=0.2, los = 70):
        """
        Load training and test datasets.
        
        :return: Loaded DataFrames for train and test datasets
        """
        df_train = pd.read_csv(self.PATH_TRAIN_DATA, names=self.DATA_SET_HEADER)
        # df_train = self.filter_by_length_of_sentence(df_train)

        df_test = pd.read_csv(self.PATH_TEST_DATA, names=self.DATA_SET_HEADER)
        if num_sample == 0:
            pass
        else:
            df_train = self.sampling_data(df_train, num_sample=num_sample)
            df_test = self.sampling_data(df_test, num_sample=int(num_sample*test_ratio))

        df_train = df_train.dropna()
        df_test = df_test.dropna()

        return df_train, df_test

    def parallel_load_data(self, num_sample=0, test_ratio=0.2, los = 70):
        """
        Load training and test datasets.
        
        :return: Loaded DataFrames for train and test datasets
        """
        df_train = dd.read_csv(self.PATH_TRAIN_DATA, names=self.DATA_SET_HEADER)
        df_test = dd.read_csv(self.PATH_TEST_DATA, names=self.DATA_SET_HEADER)

        df_train = df_train.compute()
        df_test = df_test.compute()

        # df_train = pd.read_csv(self.PATH_TRAIN_DATA, names=self.DATA_SET_HEADER)
        # # df_train = self.filter_by_length_of_sentence(df_train)

        # df_test = pd.read_csv(self.PATH_TEST_DATA, names=self.DATA_SET_HEADER)
        if num_sample == 0:
            pass
        else:
            df_train = self.sampling_data(df_train, num_sample=num_sample)
            df_test = self.sampling_data(df_test, num_sample=int(num_sample*test_ratio))

        df_train = df_train.dropna()
        df_test = df_test.dropna()

        return df_train, df_test
    
    def preprocess(self, df):
        """
        Preprocess the dataset by converting polarity to binary labels.
        
        :param df: Input DataFrame
        :return: Preprocessed DataFrame with binary labels
        """
        df['polarity'] = df['polarity'].map({1: 0, 2: 1})  # 1 for negative, 2 for positive
        return df

    def split_data(self, df):
        """
        Split the dataset into training and test sets.
        
        :param df: Input DataFrame
        :return: Training and test sets for X and y
        """
        X = df['review']  # Feature: review text
        y = df['polarity']  # Target: sentiment polarity

        return X, y

    def vectorize_text(self, X_train, X_test):
        """
        Perform TF-IDF vectorization on the training and test datasets.
        
        :param X_train: Training text data
        :param X_test: Test text data
        :return: TF-IDF transformed X_train and X_test
        """
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        return X_train_tfidf, X_test_tfidf
    
    def tokenization_and_padding(self, X_train, X_test):
        """
        Tokenization and Padding
        Pad sequences to ensure uniform input length

        :param X_train: Training text data
        :param X_test: Test text data
        :return: Padded Sequences
        """
        self.tokenizer = Tokenizer(num_words=self.max_features)
        self.tokenizer.fit_on_texts(X_train)

        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)

        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_length)
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_length)
        return X_train_pad, X_test_pad
    
    
    def _expand_contractions(self, review):
        contractions = {
            "don't": "do not", "I'm": "I am", "you're": "you are", "it's": "it is",
            "he's": "he is", "she's": "she is", "we're": "we are", "they're": "they are",
            "I've": "I have", "you've": "you have", "we've": "we have", "they've": "they have",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
            "won't": "will not", "wouldn't": "would not", "can't": "cannot", "couldn't": "could not",
            "shouldn't": "should not", "mightn't": "might not", "mustn't": "must not"
        }
        
        contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))
        
        def replace(match):
            return contractions[match.group(0)]
    
        return contractions_re.sub(replace, review)

    def _remove_stopwords(self, review):
        review = self._expand_contractions(review)
        
        word_tokens = word_tokenize(review)
        
        filtered_review = [re.sub(r'\W+$', '', word.lower()) for word in word_tokens if word.lower() not in self.stop_words]
        
        filtered_review = ' '.join(filtered_review).strip()
        filtered_review = re.sub(' +', ' ', filtered_review) 
        
        return filtered_review

    def _process_reviews_chunk(self, reviews_chunk):
        return reviews_chunk.apply(self._remove_stopwords)
    
    def remove_stopwords(self, df, col_name='review'):
        num_cores = multiprocessing.cpu_count() 
        pool = multiprocessing.Pool(num_cores)   

        review_chunks = np.array_split(df[col_name], num_cores)
        processed_chunks = pool.map(self._process_reviews_chunk, review_chunks)

        df[col_name] = pd.concat(processed_chunks)
        
        pool.close()
        pool.join()

        return df
    
    def count_sentence_words_count(self, df, col_name='review'):
        df[f'{col_name}_length'] = df[col_name].apply(lambda x: len(x.split()))
        return df
    
    def get_plot(self, df,col_name='review_length'):
        review_length_counts = df[col_name].value_counts().sort_index()

        cumulative_percentage = np.cumsum(review_length_counts.values) / np.sum(review_length_counts.values) * 100

        length_70_percent = review_length_counts.index[np.argmax(cumulative_percentage >= 70)]
        length_80_percent = review_length_counts.index[np.argmax(cumulative_percentage >= 80)]

        plt.figure(figsize=(20, 6))
        plt.bar(review_length_counts.index, review_length_counts.values, color='skyblue')
        plt.axvline(x=length_70_percent, color='red', linestyle='--', label=f'70% at {length_70_percent} words')
        plt.axvline(x=length_80_percent, color='green', linestyle='--', label=f'80% at {length_80_percent} words')
        plt.xlabel('Review Length (Number of Words)', fontsize=12)
        plt.ylabel('Number of Reviews', fontsize=12)
        plt.title('Distribution of Review Lengths in f_X_train', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()

        print('Words count - 70%: ',length_70_percent)
        print('Words count - 80%: ',length_80_percent)

processor = TextPreprocessor()
df_train, df_test = processor.parallel_load_data()

df_train_f = processor.remove_stopwords(df_train.copy())
df_test_f = processor.remove_stopwords(df_test.copy())

df_train_f = processor.filter_by_length_of_sentence(df_train_f.copy())
df_test_f = processor.filter_by_length_of_sentence(df_test_f.copy())


# Split data
X_train, y_train= processor.split_data(df_train_f)
X_test, y_test = processor.split_data(df_test_f)

X_train_tfidf, X_test_tfidf = processor.vectorize_text(X_train, X_test)

import sys
import os
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import joblib
# Importing necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import joblib
import os

OUTPUT_DIR = "Output/data_full_random"

# 저장할 디렉토리 설정 (필요 시 생성)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

best_params_path = os.path.join(OUTPUT_DIR, 'best_params.pkl')
model_path = os.path.join(OUTPUT_DIR, 'best_logistic_regression_model.pkl')

# Best parameters 불러오기
best_params = joblib.load(best_params_path)
print(f"Loaded Best Parameters: {best_params}")

# Best model 불러오기
best_model = joblib.load(model_path)
print("Loaded Best Logistic Regression Model.")

from sklearn.model_selection import GridSearchCV
# RandomizedSearchCV로 찾은 최적 범위를 기반으로 GridSearch 설정
param_grid = {
    'C': np.linspace(best_params['C'] - 1, best_params['C'] + 1, 50),  # 좁은 범위로 설정
    'solver': [best_params['solver']],  # 최적의 solver 선택
    'max_iter': [200, 300],  # 더 정밀하게
    'penalty': [best_params['penalty']]  # 최적 penalty 선택
}

grid_search = GridSearchCV(
    best_model,
    param_grid=param_grid,
    cv=3,
    verbose=1,
    n_jobs=12
)

grid_search.fit(X_train_tfidf, y_train)

print(f"Best Parameters from GridSearchCV: {grid_search.best_params_}")

model_f = LogisticRegression(**grid_search.best_params_)
model_f.fit(X_train_tfidf, y_train)

y_pred = model_f.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"Hypertuning Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

OUTPUT_DIR_GRID = "Output/data_full_random/grid"

# 저장할 디렉토리 설정 (필요 시 생성)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Best parameters 저장
best_params_path = os.path.join(OUTPUT_DIR, 'best_params.pkl')
joblib.dump(grid_search.best_params_, best_params_path)

print(f"Best parameters saved at: {best_params_path}")

# 모델 저장
model_path = os.path.join(OUTPUT_DIR, 'best_logistic_regression_model.pkl')
joblib.dump(model_f, model_path)

print(f"Best Logistic Regression model saved at: {model_path}")


