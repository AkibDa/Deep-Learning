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

for i in range(6):
  plt.subplot(2, 3, i + 1)
  plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
plt.show()