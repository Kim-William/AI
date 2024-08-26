import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from featurization import tfidf_to_sentence
import numpy as np

def evaluate(encoder, attn, decoder, featureVal, labelVal, acc_step, criterion):
    # Set the models to evaluation mode
    encoder.eval()
    attn.eval()
    decoder.eval()

    total_loss = 0.0
    bleu_scores = []

    with torch.no_grad():
        for i in range(0, len(featureVal), acc_step):
            batch_feature = featureVal[i:i + acc_step]
            batch_label = labelVal[i:i + acc_step]

            encoder_output, encoder_hidden = encoder(batch_feature)
            decoder_hidden = encoder_hidden
            attn_context = torch.zeros_like(encoder_output[0])

            batch_loss = 0.0

            for t in range(batch_label.shape[1]):
                decoder_input = batch_label[:, t]
                decoder_output, decoder_hidden, attn_context = attn(decoder_input, decoder_hidden, encoder_output, attn_context)
                output = decoder(decoder_output)

                loss = criterion(output, batch_label[:, t])
                batch_loss += loss

                _, topi = output.topk(1)
                decoder_input = topi.squeeze()

            total_loss += batch_loss.item()
            smoothing = SmoothingFunction().method1
            # Compute BLEU score for each example in the batch
            for j in range(len(batch_label)):
                reference = [tfidf_to_sentence(batch_label[j])]
                candidate = [tfidf_to_sentence(output[j])]
                bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothing)
                bleu_scores.append(bleu_score)

    avg_loss = total_loss / len(featureVal)
    avg_bleu = np.mean(bleu_scores)
    return avg_loss, avg_bleu