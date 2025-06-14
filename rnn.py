import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

# loading data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# normalising 0,255 --> 0,1
x_train, x_test = x_train / 255.0, x_test / 255.0

# model
model = keras.models.Sequential()
model.add(keras.Input(shape=(28, 28)))
model.add(keras.layers.SimpleRNN(128, return_sequences=True, activation="relu"))
model.add(keras.layers.SimpleRNN(128, return_sequences=False, activation="relu"))
model.add(keras.layers.Dense(10))

print(model.summary())
# model = keras.Sequential()
# model.add(keras.layers.Flatten(input_shape=(28, 28)))
# model.add(keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(keras.layers.Dense(10))

# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam()
metrics = ['accuracy']

model.compile(loss=loss, optimizer=optim, metrics=metrics)

# training
batch_size = 64
epochs = 5

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1)

# evaluate
model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

# predictions
probability_model = keras.models.Sequential([
  model,
  keras.layers.Softmax()
])

predictions = probability_model(x_test)
print(predictions[0])
label = np.argmax(predictions[0])
print(label)
