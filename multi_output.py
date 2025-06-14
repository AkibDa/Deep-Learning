import tensorflow as tf
import numpy as np

# Define Functional model
inputs = tf.keras.Input(shape=(28,28))
flatten = tf.keras.layers.Flatten()
dense1 = tf.keras.layers.Dense(128, activation='relu')
dense2 = tf.keras.layers.Dense(10, activation='softmax', name="category_output")
dense3 = tf.keras.layers.Dense(1, activation='sigmoid', name="leftright_output")

x = flatten(inputs)
x = dense1(x)
outputs1 = dense2(x)
outputs2 = dense3(x)

model = tf.keras.Model(inputs=inputs, outputs=[outputs1, outputs2], name="mnist_model")
model.summary()

# loss and optimizer
loss1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
loss2 = tf.keras.losses.BinaryCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam()
metrics = ['accuracy', 'accuracy']

losses = {
  'category_output': loss1,
  'leftright_output': loss2,
}

model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

# create data with 2 labels
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 0=left, 1=right
y_leftright = np.zeros(y_train.shape, dtype=np.uint8)
for idx, y in enumerate(y_train):
  if y > 5:
    y_leftright[idx] = 1
print(y_train.dtype, y_train[0:20])
print(y_leftright.dtype, y_leftright[0:20])

y = {
  'category_output': y_train,
  'leftright_output': y_leftright,
}

# training
model.fit(x=x_train, y=y, epochs=5, batch_size=64, verbose=2)

# list with 2 predictions
predictions = model.predict(x_test)
len(predictions)

predictions_category = predictions[0]
predictions_leftright = predictions[1]

pr_cat = predictions_category[0:20]
pr_l = predictions_leftright[0:20]

labels_cat = np.argmax(pr_cat, axis=1)
labels_l = np.argmax([ 1 if p >= 0.5 else 0 for p in pr_l])

print(y_test[0:20])
print(labels_cat)
print(labels_l)