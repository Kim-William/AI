import torch.nn as nn
import torch.nn.functional as F

class DecoderLSTM(nn.Module):  # Define a class DecoderLSTM that inherits from nn.Module
  def __init__(self, input_dim, hidden_dim, output_vocab_len, n_layers=1, drop_prob=0.1):  # Initialize the class with the given parameters
    super(DecoderLSTM, self).__init__()  # Call the parent class's initializer
    # Initialize class variables
    self.hidden_dim = hidden_dim
    self.output_vocab_len = output_vocab_len
    self.n_layers = n_layers
    self.drop_prob = drop_prob
    self.input_dim = input_dim
 
    # Define an embedding layer that will convert the input words (given as indices) into vectors of dimension input_dim
    self.embedding = nn.Embedding(self.output_vocab_len, self.input_dim)
    # Define a dropout layer with the given dropout probability
    self.dropout = nn.Dropout(self.drop_prob) 
    # Define an LSTM layer that takes embedded word vectors (of dimension input_dim) as input and outputs hidden states of dimension hidden_dim
    self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
    # Define a linear layer that maps from the hidden state space to the output vocabulary space
    self.classifier = nn.Linear(self.hidden_dim, self.output_vocab_len)

  def forward(self, inputs, decoder_state_vector, decoder_context_vector):  # Define the forward pass of the decoder
    # Embed the input words
    embedded = self.embedding(inputs)
    embedded = self.dropout(embedded)
    
    # Maintain the sequence length dimension when reshaping
    if len(embedded.shape) == 1:
        embedded = embedded.unsqueeze(1)

    # Pass the embedded words and the initial hidden state to the LSTM
    output, hidden = self.lstm(embedded, (decoder_state_vector, decoder_context_vector))
    # Pass LSTM outputs through a Linear layer acting as a classifier
    output = F.log_softmax(self.classifier(output.squeeze(0)), dim=1)

    return output, hidden  # Return the LSTM output and the final hidden state