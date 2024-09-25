import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

class TextPreprocessor:
    # Example usage:
    PATH_TRAIN_DATA = os.path.join(os.path.dirname(__file__), 'InputData', 'train.csv')
    PATH_TEST_DATA = os.path.join(os.path.dirname(__file__), 'InputData', 'test.csv')
    DATA_SET_HEADER = ['polarity', 'title', 'review']
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