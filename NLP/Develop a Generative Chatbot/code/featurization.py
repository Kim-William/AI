import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import torch

import nltk
nltk.download('punkt')

# Load the English language model, spacy stopwords, and punctuations
import spacy
nlp = spacy.load("en_core_web_lg")


def convert_question_to_tensor(question):
    # import dictionaries
    with open('./dictionaries/word_to_index.pkl', 'rb') as f:
        word_to_index = pickle.load(f)
    # Tokenize the question
    tokens = question.split()
    # Convert the tokens to their corresponding indices
    indices = [word_to_index.get(token, word_to_index[token]) for token in tokens]
    # Convert the indices to a tensor
    question_tensor = torch.tensor(indices, dtype=torch.long)

    return question_tensor

def convert_predictions_to_response(top_indices):
    # import dictionaries
    with open('./dictionaries/index_to_word.pkl', 'rb') as f:
        index_to_word = pickle.load(f)

    # Convert the indices to their corresponding words
    words = [[index_to_word[index] for index in sentence if index in index_to_word] for sentence in top_indices]

    # Join the words into a response
    responses = [" ".join(sentence) for sentence in words]
    print("..............................")
    print(responses)
    return responses


def tfidf_to_sentence(tfidf_mtx, index_to_word):
    # Convert the sparse matrix to a dense numpy array
    dense_vector = np.array(tfidf_mtx)
    # Get the indices of the non-zero elements
    word_indices = np.flatnonzero(dense_vector)

    # Convert the indices to words using the index_to_word dictionary
    words = [index_to_word[index] for index in word_indices]

    return words

def featurization_TFIDF(a, b):
    # Initialize a TfidfVectorizer object with a specific tokenizer, minimum document frequency, and ngram range
    tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer, min_df=5, ngram_range=(1,3))
    # Fit the TfidfVectorizer to the training data
    tfidf_vector.fit(a)

    # Create the word_to_index dictionary
    word_to_index = tfidf_vector.vocabulary_

    # Create the index_to_word dictionary
    index_to_word = {index: word for word, index in word_to_index.items()}

    # Save the dictionaries to a file
    with open('./dictionaries/word_to_index.pkl', 'wb') as f:
        pickle.dump(word_to_index, f)

    with open('./dictionaries/index_to_word.pkl', 'wb') as f:
        pickle.dump(index_to_word, f)
    
    # Transform training and test features with the learned vocabulary, and return them for training
    A = tfidf_vector.transform(a)
    B = tfidf_vector.transform(b)
    return A, B

def spacy_tokenizer(text):
    """ 
    This function takes a string of text as input and returns a list of lemmatized tokens (words),
    where each token is in lower case, non-alphabetic characters and stop words are removed.
    """
    # Remove non-alphabetic characters from the text using a regular expression
    cleaned_text = re.sub(r"[^a-zA-Z\s]", "", text)
    # For each token in the processed text, check if it is an alphabetic word and not a stop word
    # If it is, lemmatize the token and convert it to lower case, then add it to the list of tokens
    tokens = [token.lemma_.lower() for token in nlp(cleaned_text) if token.is_alpha and not token.is_stop]
    # Return the list of tokens
    return tokens