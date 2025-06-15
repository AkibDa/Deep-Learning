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
len(counter)




