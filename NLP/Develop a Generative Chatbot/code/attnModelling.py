from featurization import featurization_TFIDF
from encoder import EncoderLSTM
from attention import AttnDecoderLSTM
from earlyStopper import EarlyStopper

import logging
import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score

def attnModel(train, test):
    
    trainQ = train['encoder']
    testQ = test['encoder']
    trainA = train['decoder']
    testA = test['decoder']

    early_stopping = EarlyStopper(patience=3)
    featureTrain, featureTest = featurization_TFIDF(trainQ, testQ)
    labelTrain, labelTest = featurization_TFIDF(trainA, testA)

    input_dim = len(featureTrain)
    hidden_dim = 256
    lr = 0.001
    EPOCHS = 10
    teacher_forcing_prob = 0.5

    encoder = EncoderLSTM(len(trainQ), input_dim, hidden_dim)
    decoder = AttnDecoderLSTM(input_dim, hidden_dim, len(trainA))

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

    # Define learning rate schedulers
    encoder_scheduler = StepLR(encoder_optimizer, step_size=30, gamma=0.1)
    decoder_scheduler = StepLR(decoder_optimizer, step_size=30, gamma=0.1)

    encoder.train()
    decoder.train()

    tk0 = range(1,EPOCHS+1)

    for epoch in tk0:
        avg_loss = 0.
        total_bleu_score = 0.
        total_accuracy = 0.
        tk1 = enumerate(featureTrain)

        for i, sentence in tk1:

            loss = 0.

            #initialise encoder state vector and cell state vector
            h = encoder.init_hidden()
            encoder_state_vector = h[0]
            encoder_cell_vector = h[0]

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            inp = featureTrain[i]

            if (i % 100) == 0:
                logging.info(f"inp: {i}, epoch: {epoch}")

            # Convert the numpy array to a torch Tensor
            tensor = torch.from_numpy(inp).long()
            tensor = tensor.view(1, -1)
            tensor = pad_sequence(tensor, batch_first=True)
            encoder_outputs, h = encoder(tensor, encoder_state_vector, encoder_cell_vector)
            encoder_outputs = encoder_outputs.view(1, -1, 100)
            encoder_outputs = encoder_outputs.transpose(1, 2)
            output = []
            teacher_forcing = True if random.random() < teacher_forcing_prob else False
            
            for ii in range(len(labelTrain[i])):
                decoder_output, decoder_hidden = decoder(tensor, h, encoder_outputs)
                dense_array = labelTrain[i][ii]
                # Get the index value of the word with the highest score from the decoder output
                decoder_output = decoder_output.view(-1)
                top_value, top_index = decoder_output.topk(1)
                if teacher_forcing:            
                    decoder_input = torch.tensor(dense_array)
                else:
                    decoder_input = torch.tensor([top_index.item()])
                
                output.append(top_index.item())
                # Calculate the loss of the prediction against the actual word
                loss += F.nll_loss(decoder_output.view(1,-1), torch.tensor([dense_array]).long())

            # Calculate BLEU score
            bleu_score = sentence_bleu([labelTrain[i]], output)
            # # Calculate accuracy - requires the same type... -.-
            output = [int(x) for x in output]
            labelTrain[i] = [int(x) for x in labelTrain[i]]
            accuracy = accuracy_score(labelTrain[i], output)

            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)

            encoder_optimizer.step()
            decoder_optimizer.step()
            # Step the learning rate scheduler
            encoder_scheduler.step()
            decoder_scheduler.step()

            avg_loss += loss.item()/len(trainQ)
            total_bleu_score += bleu_score
            total_accuracy += accuracy

            avg_bleu_score = total_bleu_score / len(trainQ)
            avg_accuracy = total_accuracy / len(trainQ)
 
        logging.info(f"Epoch: {epoch}, Loss: {avg_loss}, BLEU Score: {avg_bleu_score}, Accuracy: {avg_accuracy}")

        if early_stopping.early_stop(avg_loss):
            break
    
    # Save model after every epoch (Optional)
    torch.save({"encoder":encoder.state_dict(),"decoder":decoder.state_dict(),"e_optimizer":encoder_optimizer.state_dict(),"d_optimizer":decoder_optimizer},"model_enc_att.pt")