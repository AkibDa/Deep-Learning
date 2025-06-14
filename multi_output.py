import tensorflow as tf
import numpy as np

# Define Functional model
inputs = tf.keras.Input(shape=(28,28))
flatten = tf.keras.layers.Flatten()
dense1 = tf.keras.layers.Dense(128, activation='relu')
dense2 = tf.keras.layers.Dense(10, activation='softmax', name="category_output")
dense3 = tf.keras.layers.Dense(1, activation='sigmoid', name="leftright_output")

x = flatten(inputs)
x = dense1(x)
outputs1 = dense2(x)
outputs2 = dense3(x)

model = tf.keras.Model(inputs=inputs, outputs=[outputs1, outputs2], name="mnist_model")
model.summary()

# loss and optimizer
loss1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
loss2 = tf.keras.losses.BinaryCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam()
metrics = ['accuracy']

losses = {
  'category_output': loss1,
  'leftright_output': loss2,
}

model.compile(optimizer=optimizer, loss=losses, metrics=metrics)