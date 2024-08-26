import pandas as pd
import torch
from encoder import EncoderLSTM
from decoder import DecoderLSTM
from featurization import featurization_TFIDF
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

from featurization import convert_predictions_to_response

def predict(data_size):

    validation_df = pd.read_csv("./preprocessed/validation_cleaned.csv").head(data_size)

    valQ = validation_df['encoder']
    valA = validation_df['decoder']
    q, a = featurization_TFIDF(valQ, valA)
    
    inpQ = q.shape[0]
    inpA = a.shape[0]
    valQ_tensor = torch.tensor(inpQ).long()
    valA_tensor = torch.tensor(inpA).long()
    hidden_dim = 512

    # Load the state dictionaries
    state_dicts = torch.load('./models/model_enc_dec_0833.pt')

    # Create the models and optimizers
    encoder = EncoderLSTM(len(valQ), inpQ, hidden_dim)
    decoder = DecoderLSTM(inpQ, hidden_dim, len(valA))

    encoder_optimizer = torch.optim.Adam(encoder.parameters())
    decoder_optimizer = torch.optim.Adam(decoder.parameters())

    # Load the state dictionaries into the models and optimizers
    encoder.load_state_dict(state_dicts['encoder'])
    decoder.load_state_dict(state_dicts['decoder'])
    encoder_optimizer.load_state_dict(state_dicts['e_optimizer'])
    decoder_optimizer.load_state_dict(state_dicts['d_optimizer'])

    h = encoder.init_hidden()
    encoder_state_vector = h[0].expand(1, len(valQ), -1)
    encoder_cell_vector = h[1].expand(1, len(valQ), -1)
    # Set the models to evaluation mode
    encoder.eval()
    decoder.eval()

    encoder_outputs, encoder_hidden = encoder(valQ_tensor, encoder_state_vector, encoder_cell_vector)

    decoder_hidden = encoder_hidden
    predictions = torch.zeros(valQ_tensor.size(0), decoder.output_vocab_len)

    decoder_output, decoder_hidden = decoder(valA_tensor, encoder_hidden[0], encoder_hidden[1])
    predictions = decoder_output

    # Find the indices of the maximum values
    top_indices = predictions.argmax(dim=-1).tolist()

    # Convert indices to words
    predicted_words = convert_predictions_to_response(top_indices)
    print(predicted_words)


    # Convert the actual responses to a list of words
    actual_words = [sentence.split() for sentence in valA]

    # Calculate the BLEU score
    bleu_score = corpus_bleu([[sentence] for sentence in actual_words], predicted_words)

    print(f'BLEU score: {bleu_score}')