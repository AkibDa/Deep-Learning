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

# functional API
inputs = keras.Input(shape=(28, 28))

flatten = keras.layers.Flatten()
dense1 = keras.layers.Dense(64, activation='relu')
dense2 = keras.layers.Dense(10)

x = flatten(inputs)
x = dense1(x)
outputs = dense2(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='functional_model')

print(model.summary())