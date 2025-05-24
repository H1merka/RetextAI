# RetextAI #

## What is this? ##
This repository is a telegram chatbot that uses artificial intelligence to paraphrase texts written in Russian.


----------


### Installing dependencies ###


To install the necessary dependencies, run the installation command using the package manager `pip`:

    pip install -r requirements.txt


For the correct operation of the installed modules, the installed programming language `Rust` is required, as well as the package manager `Cargo`.


It is also worth considering that the bot will require a version `3.9<=Python<=3.12`.


### Usage ###


This program uses a pre-trained language model `cointegrated/rut5-base-paraphraser`: [link](https://huggingface.co/cointegrated/rut5-base-paraphraser).


The token for the bot was obtained from `BotFather`.


The `paraphrase` function is designed to paraphrase text using a machine learning model. `text` is the source text to be paraphrased, `sequences` is the number of paraphrasing options to be returned, `beams` is the number of beams (beam search) to find the best result; the larger the beams, the better the quality, but the longer the calculations, `grams` is the size of the n—grams for to prevent repetition; this parameter tells the model how many previous words to consider when generating the next word. `do_sample` is a flag that includes randomness in the selection of the next word.

    paraphrase(text, sequences, beams=15, grams=4, do_sample=True)


The `start` function is an asynchronous function designed to process the initial message from the user in the bot, responsible for greeting the user and providing instructions on how to use the bot. `update` is an object containing information about the message received from the user, `context` is a context containing information about the current status of the conversation with the user.

    start(update: Update, context: ContextTypes.DEFAULT_TYPE)


The `help_command` function is an asynchronous function designed to process the `/help` command in the bot. It is responsible for providing the user with instructions on how to use the bot.

    help_command(update: Update, context: ContextTypes.DEFAULT_TYPE)


The `handle_message` function handles incoming messages from users in the bot. It manages the state of the dialog and performs various actions depending on the current state.

    handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE)


----------


## The development team ##
GitHub link: [link](https://github.com/H1merka)