# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
from encoder import EncoderLSTM
from decoder import DecoderLSTM
from featurization import convert_predictions_to_response, convert_question_to_tensor

from botbuilder.dialogs import (
    ComponentDialog,
    WaterfallDialog,
    WaterfallStepContext,
    DialogTurnResult,
)
from botbuilder.dialogs.prompts import (
    TextPrompt,
    PromptOptions
)
from botbuilder.core import MessageFactory, UserState

class UserProfileDialog(ComponentDialog):
    def __init__(self, user_state: UserState):
        super(UserProfileDialog, self).__init__(UserProfileDialog.__name__)

        self.user_profile_accessor = user_state.create_property("UserProfile")

        self.add_dialog(
            WaterfallDialog(
                WaterfallDialog.__name__,
                [
                    self.question_step
                ],
            )
        )
        self.add_dialog(TextPrompt(TextPrompt.__name__))
        self.initial_dialog_id = WaterfallDialog.__name__

    async def question_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        # Load the state dictionaries
        state_dicts = torch.load('./models/model_enc_dec_1408.pt')
        
        user_question = step_context.result
        hidden_dim = 256

        # Convert the user's question to a tensor that your model can understand
        # This will depend on how you've processed your data during training
        question_tensor = convert_question_to_tensor(user_question)
        # Create the models and optimizers
        encoder = EncoderLSTM(len(user_question), len(question_tensor), hidden_dim)
        decoder = DecoderLSTM(len(question_tensor), hidden_dim, len(user_question))
        
        encoder_optimizer = torch.optim.Adam(encoder.parameters())
        decoder_optimizer = torch.optim.Adam(decoder.parameters())

        # Load the state dictionaries into the models and optimizers
        encoder.load_state_dict(state_dicts['encoder'])
        decoder.load_state_dict(state_dicts['decoder'])
        encoder_optimizer.load_state_dict(state_dicts['e_optimizer'])
        decoder_optimizer.load_state_dict(state_dicts['d_optimizer'])

        h = encoder.init_hidden()
        encoder_state_vector = h[0].expand(1, len(user_question), -1)
        encoder_cell_vector = h[1].expand(1, len(user_question), -1)

        # Pass the question through the encoder
        encoder_outputs, encoder_hidden = encoder(question_tensor, encoder_state_vector, encoder_cell_vector)

        # Initialize the decoder's hidden state with the encoder's hidden state
        decoder_hidden = encoder_hidden

        # Initialize an empty tensor to hold the decoder's predictions
        predictions = torch.zeros(question_tensor.size(0), decoder.output_vocab_len)

        # Decode the encoder's output
        for t in range(question_tensor.size(0)):
            decoder_output, decoder_hidden = decoder(decoder_hidden, encoder_outputs)
            predictions[t] = decoder_output

        # Convert the predictions tensor to a human-readable response
        response = convert_predictions_to_response(predictions)

        # Return the response
        return await step_context.prompt(
            TextPrompt.__name__,
            PromptOptions(prompt=MessageFactory.text(response)),
        )
