import tensorflow as tf

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

model.add(tf.keras.layers.Dense(5))

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
metrics = ['accuracy']

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)