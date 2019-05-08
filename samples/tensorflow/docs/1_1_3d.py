import numpy as np
import tensorflow as tf

"""
http://www.tensorfly.cn/tfdoc/get_started/introduction.html
"""

x_data = np.float32(np.random.rand(100, 2))
y_data = np.matmul(x_data, [[0.1], [0.2]]) + 0.3

W = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(x_data, W) + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(0, 201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(W), sess.run(b), sess.run(loss))
