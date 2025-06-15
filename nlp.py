import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

df = pd.read_csv('data/tweets.csv')
print(df.head())
print(df.shape)
print((df.target == 1).sum()) # Disaster
print((df.target == 0).sum()) # No Disaster

# Preprocessing
import re
import string

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r' ', text)