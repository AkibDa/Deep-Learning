import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight','Acceleration','Model Year','Origin']

dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

print(dataset.tail())

# clean data
dataset = dataset.dropna()

# convert categorical 'Origin' data into one-hot data
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1
dataset['Europe'] = (origin == 2) * 1
dataset['Japan'] = (origin == 3) * 1

print(dataset.tail())

# split the data into train and test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

print(dataset.shape, train_dataset.shape, test_dataset.shape)
print(train_dataset.describe().transpose())

# split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

def plot(feature, x=None, y=None):
  plt.figure(figsize=(10, 8))
  plt.scatter(train_features[feature], train_labels, label='Data')
  if x is not None and y is not None:
    plt.plot(x, y, color='k', label='Prediction')
  plt.xlabel(feature)
  plt.ylabel('MPG')
  plt.legend()

plot('Horsepower')
plot('Weight')

# Normalise the data
print(train_dataset.describe().transpose()[['mean', 'std']])

# Normalization
normalizer = layers.Normalization()

# Adapt to the data
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

# When the layer is called it returns the input data, with each features independently normalised:
# (input-mean)/stddev
first = np.array(train_features[:1])
print('First example: ', first)
print('Normalized: ',normalizer(first).numpy())

# Regression
#  1. Normalize the input horsepower
#  2. Apply a linear transformation (y=m*x+b) to produce 1 output using layers.Dense

feature = 'Horsepower'
single_feature = np.array(train_features[[feature]])
print(single_feature.shape, train_features.shape)

single_feature_normalizer = layers.Normalization()
single_feature_normalizer.adapt(single_feature)

single_feature_model = keras.models.Sequential([
  single_feature_normalizer,
  layers.Dense(units=1), # Linear Regression Model
])
print(single_feature_model.summary())

# loss and optimizer
loss = keras.losses.MeanAbsoluteError()
optimizer = keras.optimizers.Adam()

single_feature_model.compile(loss=loss, optimizer=optimizer)

# training the model
history = single_feature_model.fit(train_features[feature],
                                   train_labels,
                                   epochs=100,
                                   verbose=1,
                                   validation_split=0.2)

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 25])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
plot_loss(history)

