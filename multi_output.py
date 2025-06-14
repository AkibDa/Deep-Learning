import tensorflow as tf
import numpy as np

# Define Functional model
inputs = tf.keras.Input(shape=(28,28))
flatten = tf.keras.layers.Flatten()
dense1 = tf.keras.layers.Dense(128, activation='relu')
dense2 = tf.keras.layers.Dense(10, activation='softmax', name="category_output")
dense3 = tf.keras.layers.Dense(1, activation='sigmoid', name="leftright_output")