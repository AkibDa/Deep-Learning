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




