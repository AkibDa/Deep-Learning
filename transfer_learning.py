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