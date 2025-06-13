import tensorflow as tf

#rank 1 tensor
x = tf.constant([1,2,3])
print(x)

#rank 2 tensor
y = tf.constant([[1,2,3],[4,5,6]])
print(y)

z = tf.ones([2,2])
print(z)
