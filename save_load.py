import tensorflow as tf
from tensorflow import keras
import numpy as np

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential([
  keras.layers.Flatten(input_shape=(28,28)),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(10),
])

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam()
metrics = [keras.metrics.SparseCategoricalAccuracy()]

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

model.fit(x_train, y_train, epochs=5, batch_size=64, shuffle=True, verbose=2)

print("Evaluate: ")
model.evaluate(x_test, y_test, verbose=2)

# Save the model
model.save('neural_net.h5')

# Loading the model
new_model = keras.models.load_model('neural_net.h5')
new_model.evaluate(x_test, y_test, verbose=2)