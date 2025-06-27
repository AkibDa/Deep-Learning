import os
import math
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = 'data/marvel/'
names = ['Spiderman','Venom','Aunt May','Gwen','Mysterio']

tf.random.set_seed(1)

# Read information about dataset
if not os.path.isdir(BASE_DIR + 'train/'):
  for name in names:
    os.makedirs(BASE_DIR + 'train/' + name)
    os.makedirs(BASE_DIR + 'test/' + name)
    os.makedirs(BASE_DIR + 'val/' + name)

# Total number of classes in the dataset
orig_folders = ['0001/', '0002/', '0003/', '0004/', '0005/']
for folder_idx, folder in enumerate(orig_folders):
  files = os.listdir(BASE_DIR + folder)
  number_of_images = len([name for name in files])
  n_train = int((number_of_images * 0.6) + 0.5)
  n_valid = int((number_of_images * 0.25) + 0.5)
  n_test = number_of_images - n_train - n_valid
  print(number_of_images, n_train, n_valid, n_test)
  for idx, file in enumerate(files):
    file_name = BASE_DIR + folder +  file
    if idx < n_train:
      shutil.move(file_name, BASE_DIR + 'train/' + names[folder_idx])
    elif idx < n_train + n_valid:
      shutil.move(file_name, BASE_DIR + 'val/' + names[folder_idx])
    else:
      shutil.move(file_name, BASE_DIR + 'test/' + names[folder_idx])

train_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
valid_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_batches = train_gen.flow_from_directory(
  BASE_DIR + 'train/',
  target_size=(256, 256),
  batch_size=4,
  class_mode='sparse',
  shuffle=True,
  color_mode='rgb',
  classes=names,
)
valid_batches = valid_gen.flow_from_directory(
  BASE_DIR + 'val/',
  target_size=(256, 256),
  batch_size=4,
  class_mode='sparse',
  shuffle=False,
  color_mode='rgb',
  classes=names,
)
test_batches = test_gen.flow_from_directory(
  BASE_DIR + 'test/',
target_size=(256, 256),
  batch_size=4,
  class_mode='sparse',
  shuffle=False,
  color_mode='rgb',
  classes=names,
)

def show(batch, pred_labels=None):
  plt.figure(figsize=(10, 10))
  for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(batch[0][i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, which is why you need the extra index
    lbl = names[int(batch[1][i])]
    if pred_labels is not None:
      lbl += '/Pred:' + names[int(pred_labels[i])]
    plt.xlabel(lbl)
  plt.show()

test_batch = test_batches[0]
train_batch = train_batches[0]

show(test_batch)
show(train_batch)

model = keras.models.Sequential([
  keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 3)),
  keras.layers.MaxPooling2D(2, 2),
  keras.layers.Conv2D(64, (3, 3), activation='relu'),
  keras.layers.MaxPooling2D(2, 2),
  keras.layers.Flatten(),
  keras.layers.Dense(64, activation='relu'),
  keras.layers.Dense(5),
])

print(model.summary())

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam()
metrics = ['accuracy']

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# training
epochs = 30

# callbacks
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2)

history = model.fit(train_batches, epochs=epochs, callbacks=[early_stopping], validation_data=valid_batches, verbose=2)

model.evaluate(test_batches, verbose=2)

predictions = model.predict(test_batches)
predictions = tf.nn.softmax(predictions)
labels = np.argmax(predictions, axis=1)

print(test_batches[0][1])
print(labels[0:4])

show(test_batches[0], labels[0:4])