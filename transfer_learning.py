import tensorflow as tf

vgg_model = tf.keras.applications.VGG19()
print(type(vgg_model))
print(vgg_model.summary())