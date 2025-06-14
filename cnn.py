import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  layers
import matplotlib.pyplot as plt

cifar10 = keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(train_images.shape)

train_images = train_images / 255.0
test_images = test_images / 255.0