import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

df = pd.read_csv('data/tweets.csv')
print(df.head())
print((df.target == 1).sum()) # Disaster
print((df.target == 0).sum()) # No Disaster

# Preprocessing
import re
import string

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r' ', text)

def remove_punctuation(text):
  translator = str.maketrans('', '', string.punctuation)
  return text.translate(translator)

pattern = re.compile(r'https?://\S+|www\.\S+')
for t in df.text:
  matches = pattern.findall(t)
  for match in matches:
    print(t)
    print(match)
    print(pattern.sub(r"", t))
  if len(matches) > 0:
    break

df['text'] = df.text.map(remove_URL)
df['text'] = df.text.map(remove_punctuation)

# remove stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

def remove_stopwords(text):
  filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
  return ' '.join(filtered_words)

df['text'] = df.text.map(remove_stopwords)

from collections import Counter

def counter_words(text_col):
  count = Counter()
  for text in text_col.values:
    for word in text.split():
      count[word] += 1
  return count

counter = counter_words(df.text)
num_unique_words = len(counter)

# Split dataset into training and validation sets
train_size = int(df.shape[0] * 0.8)

train_df = df[:train_size]
val_df = df[train_size:]

# Split text and label
train_sentence = train_df.text.to_numpy()
val_sentence = val_df.text.to_numpy()
train_labels = train_df.target.to_numpy()
val_labels = val_df.target.to_numpy()

# Tokenize
from tensorflow.keras.preprocessing.text import Tokenizer

# vectorise a text corpus by turing each text into a sequence of integers
tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(train_sentence)

word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentence)
val_sequences = tokenizer.texts_to_sequences(val_sentence)

print(train_sentence[10:15])
print(val_sentence[10:15])

# Pad the sequence to have the same length
from tensorflow.keras.preprocessing.sequence import pad_sequence

# Max number of words in a sequence
max_length = 20

train_padded = pad_sequence(train_sequences, maxlen=max_length, padding='post', truncating='post')
val_padded = pad_sequence(val_sequences, maxlen=max_length, padding='post', truncating='post')

print(train_sequences[10])
print(train_sentence[10])
print(train_padded[10])

# Check reversing the indices
# Flip (key, value)
reverse_word_index = dict([(idx, word) for (word, idx) in word_index.items()])
print(reverse_word_index)

def decode(sequence):
  return ' '.join([reverse_word_index.get(idx, "?") for idx in sequence])

decoded_text = decode(train_sentence[10])
print(train_sentence[10])
print(decoded_text)
