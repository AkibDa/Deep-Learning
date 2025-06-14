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

BASE_DIR = 'data/star-wars/'
names = ['YODA', 'LUKE SKYWALKER', 'R2-D2', 'MACE WINDU', 'GENERAL GRIEVOUS']

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

# Generate batches of tensor image data with real-time data argumentation
# preprocessing_function
# rescale = 1./255 -> [0,1]

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
  shuffle=True,
  color_mode='rgb',
  classes=names,
)
test_batches = test_gen.flow_from_directory(
  BASE_DIR + 'test/',
target_size=(256, 256),
  batch_size=4,
  class_mode='sparse',
  shuffle=True,
  color_mode='rgb',
  classes=names,
)

