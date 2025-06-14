import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  layers
import matplotlib.pyplot as plt

cifar10 = keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(train_images.shape)

train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', ' dog', 'frog', 'horse', 'ship', 'truck']

# def show():
#   plt.figure(figsize=(10, 10))
#   for i in range(16):
#     plt.subplot(4, 4, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i][0]])
#   plt.show()
#
# show()

# Model
model = keras.models.Sequential()