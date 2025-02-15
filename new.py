import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import sys
print(sys.executable)

# The Natural Language Toolkit (NLTK) is a powerful and widely used Python library for natural language processing (NLP). It provides a #suite of libraries and tools to help process, analyze, and manipulate human language data. NLTK is widely used in academia, research, #and industry for tasks like text analysis, sentiment analysis, machine translation, and more.
#NLTK provides a vast collection of functionalities that help with:
# Tokenization : Splitting text into words or sentences
# Stemming & Lemmatization : Reducing words to their root form
# POS (Part-of-Speech) Tagging : Identifying nouns, verbs, adjectives, etc.
# Parsing & Syntax Analysis : Analyzing sentence structure
# Named Entity Recognition (NER) : Identifying entities like names, places, dates
# Text Classification : Machine learning models for categorizing text
# Sentiment Analysis : Determining the emotion behind text
# Corpora & Lexical Resources : Access to pre-built datasets and dictionaries """

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ','] # These characters will be ignored while reading the json file contents 

# words: Stores all the words (tokens) extracted from the training dataset.
# classes: Stores unique intent tags (categories) found in the dataset.
# documents: Stores tuples of (tokenized sentence, corresponding intent tag).
# ignoreLetters: List of punctuation marks that should be ignored during processing.

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern) 
        # The word_tokenize() function in NLTK splits a sentence or text into individual words (tokens). It considers punctuation marks, contractions, and special characters properly.
        #Tokenizes the sentence into words.
        #Example: "Hello! How are you?" → ['Hello', '!', 'How', 'are', 'you', '?']
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        #Adds tokenized words to the words list.
        #Stores a tuple (tokenized sentence, intent tag) in documents.
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        #Ensures unique intent tags are stored in classes.

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))
# Lemmatization: Converts words to their base (root) form.
# Ignores punctuation: Filters out unwanted symbols like ['?', '!', '.', ','].

classes = sorted(set(classes))
#set(words): Removes duplicates.
#sorted(...): Sorts words alphabetically for consistency.

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
# pickle.dump(object, file) serializes (saves) Python objects to a file.
# 'wb' mode stands for write binary, meaning the file is saved in a binary format.

#words.pkl → Stores the list of unique, lemmatized words.
#classes.pkl → Stores the list of unique intent categories.

training = []
outputEmpty = [0] * len(classes)
# training: A list that will store feature vectors (bag-of-words) along with their corresponding intent labels.
# outputEmpty: A list of zeros with the same length as classes. This represents a one-hot encoding format for intent classification.

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words: bag.append(1) if word in wordPatterns else bag.append(0)
    #Checks if each word in words exists in wordPatterns.
    #If it does, appends 1; otherwise, appends 0
    #Example:
    #words = ["hello", "goodbye", "thanks"]
    #wordPatterns = ["hello"]
    # Bag-of-Words Representation
    #bag = [1, 0, 0]

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1     #Finds the index of the intent tag in classes and sets it to 1.
    training.append(bag + outputRow)              #Combines bag-of-words (bag) and one-hot encoding (outputRow) into a single vector.

random.shuffle(training)                        #Shuffles the training data to avoid bias.
training = np.array(training)                   #Converts it into a NumPy array for better performance.

trainX = training[:, :len(words)]               #trainX contains the Bag-of-Words representations.
trainY = training[:, len(words):]               #trainY contains the one-hot encoded intent labels.

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))                                           
# ReLU (Rectified Linear Unit) helps avoid vanishing gradients.
model.add(tf.keras.layers.Dropout(0.5))
# Drops 50% of neurons randomly during training.
# Helps prevent overfitting, making the model generalize better.
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))
# softmax -> Converts outputs into probabilities for each intent.

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# SGD (Stochastic Gradient Descent) is used as the optimizer.
# Learning Rate (0.01): Determines how much the model updates in each step.
# Momentum (0.9): Helps accelerate learning by considering past gradients.
# Nesterov=True: Uses Nesterov Accelerated Gradient (NAG), improving convergence speed

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
# epochs=200: Trains the model for 200 iterations over the dataset.
# batch_size=5: Updates weights after every 5 samples.
# verbose=1: Displays training progress.

model.save('chatbot1.h5', hist)
print('Executed')