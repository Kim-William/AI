import torch.nn as nn
import torch
import torch.nn.functional as F

class AttnDecoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(AttnDecoderLSTM, self).__init__()

        self.hidden_dim = hidden_dim  # Set the hidden dimension
        self.embedding = nn.Embedding(output_dim,hidden_dim)  # Embedding layer
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)  # Linear layer for attention
        self.v = nn.Parameter(torch.rand(hidden_dim))  # Parameter for attention
        self.lstm = nn.LSTM(input_dim, hidden_dim)  # LSTM layer
        self.out = nn.Linear(hidden_dim, output_dim)  # Output layer
    
    def forward(self, inputs, hidden, encoder_outputs):
        embedded = self.embedding(inputs)  # Embed the inputs
        embedded = self.dropout(embedded)  # Apply dropout

        # Squeeze the hidden state and expand it to the size of the embedded input
        hidden_squeezed = hidden[0].squeeze(0)
        hidden_expanded = hidden_squeezed.expand(embedded[0].size())

        # Calculate the attention weights
        attn_weights = torch.sum(self.v*torch.tanh(self.attn(torch.cat((embedded[0], hidden_expanded), dim=1))), dim=1)
        attn_weights = F.softmax(attn_weights, dim=0).unsqueeze(0).unsqueeze(0)  # Apply softmax to the attention weights

        context = torch.matmul(attn_weights, encoder_outputs)  # Multiply the attention weights with the encoder outputs to get the context
        context = context.expand_as(embedded)  # Expand the context to the size of the embedded input
        rnn_input = torch.cat((embedded,context), 2)  # Concatenate the embedded input and the context

        output, hidden = self.lstm(rnn_input, hidden)  # Pass the concatenated input through the LSTM
        output = F.log_softmax(self.out(output[0]), dim = 1)  # Apply log softmax to the output

        return output  # Return the output
