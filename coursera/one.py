import tensorflow as tf
from tensorflow.keras import layers, models

# Load the Fashion MNIST dataset
(train_images, train_activityels), (test_images, test_activityels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize the pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0