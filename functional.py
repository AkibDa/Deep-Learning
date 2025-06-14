import tensorflow as tf
from tensorflow import  keras
import numpy as np

# model : sequential : one input, one output
model = keras.models.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(10),
])

print(model.summary())
