import tensorflow as tf

# tensor are like matrices

#rank 1 tensor
# x = tf.constant([1,2,3])
# print(x)

#rank 2 tensor
# y = tf.constant([[1,2,3],[4,5,6]])
# print(y)
#
# z = tf.ones([2,2])
# print(z)
#
# w = tf.random.normal([2,2], mean=0, stddev=1)
# print(w)
#
# x = tf.cast([1,2,3],tf.float32)
# print(x)

# all operation in tensor are elementwise

x = tf.constant([1,2,3])
y = tf.constant([4,5,6])

z = tf.add(x,y)
print(z)

z = tf.tensordot(x,y,[1,1])
print(z)