# teamA

A chatbot that knows the answer to all your questions.

This bot has been created using [Bot Framework](https://dev.botframework.com), it shows how to create a simple bot that accepts input from the user replies to it.

## Prerequisites

- Run `pip install -r requirements.txt` to install all dependencies
- Run `python app.py`

## Train the model

- Presonalize your settings in `starter.py`: data size to process, model to train and use to predict
- Run `python starter.py`

## Testing the bot using Bot Framework Emulator

[Bot Framework Emulator](https://github.com/microsoft/botframework-emulator) is a desktop application that allows bot developers to test and debug their bots on localhost or running remotely through a tunnel.

- Install the Bot Framework Emulator version 4.3.0 or greater from [here](https://github.com/Microsoft/BotFramework-Emulator/releases)

### Connect to the bot using Bot Framework Emulator

- Launch Bot Framework Emulator
- Enter a Bot URL of `http://localhost:3978/api/messages`