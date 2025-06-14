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
model.add(layers.Conv2D(32, (3, 3), strides=(1,1), padding='valid', activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

print(model.summary())
# import sys; sys.exit()

# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam()
metrics = ['accuracy']

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# training
batch_size = 64
epochs = 5

model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, verbose=2)

# evaluate
model.evaluate(test_images, test_labels, batch_size=batch_size, verbose=2)