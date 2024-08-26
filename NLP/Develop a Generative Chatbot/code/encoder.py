import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLSTM(nn.Module):  # Define a class EncoderLSTM that inherits from nn.Module
  def __init__(self, vocab_len, input_dim, hidden_dim, n_layers=1, drop_prob=0):  # Initialize the class with the given parameters
    super(EncoderLSTM, self).__init__()  # Call the parent class's initializer

    # Initialize class variables
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers

    # Define a dropout layer with the given dropout probability
    self.dropout = nn.Dropout(drop_prob)

    # Define an embedding layer that will convert the input words (given as indices) into vectors of dimension input_dim
    self.embedding = nn.Embedding(vocab_len, input_dim)
    
    # Define an LSTM layer that takes embedded word vectors (of dimension input_dim) as input and outputs hidden states of dimension hidden_dim
    self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
 
  def forward(self, inputs, encoder_state_vector, encoder_cell_vector):  # Define the forward pass of the encoder
    embedded = self.embedding(inputs)  # Embed the input words
    # Pass the embedded words and the initial hidden state to the LSTM
    output, hidden = self.lstm(embedded, (encoder_state_vector, encoder_cell_vector))
    output = self.dropout(output)  # Apply dropout to the LSTM output (optional)
    return output, hidden  # Return the LSTM output and the final hidden state
 
  def init_hidden(self, batch_size=1):  # Define a method to initialize the hidden state
    # The hidden state is a tuple of two tensors: the cell state and the hidden state
    # Both are zero tensors of shape (n_layers, batch_size, hidden_dim)
    return (torch.zeros(self.n_layers, batch_size, self.hidden_dim),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim))
