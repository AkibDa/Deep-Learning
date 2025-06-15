import tensorflow as tf
import matplotlib.pyplot as plt

BASE_DIR = 'data/star-wars/'
names = ['YODA', 'LUKE SKYWALKER', 'R2-D2', 'MACE WINDU', 'GENERAL GRIEVOUS']

vgg_model = tf.keras.applications.VGG19()
print(type(vgg_model))
print(vgg_model.summary())

model = tf.keras.models.Sequential()
for layer in vgg_model.layers[0:-1]:
  model.add(layer)
print(model.summary())

for layer in model.layers:
  layer.trainable = False
print(model.summary())

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

model.add(tf.keras.layers.Dense(5))

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
metrics = ['accuracy']

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

preprocess_input = tf.keras.applications.vgg19.preprocess_input

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = preprocess_input)
valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = preprocess_input)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = preprocess_input)

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

# training
epochs = 30

# callbacks
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2)

history = model.fit(train_batches, epochs=epochs, callbacks=[early_stopping], validation_data=valid_batches, verbose=2)