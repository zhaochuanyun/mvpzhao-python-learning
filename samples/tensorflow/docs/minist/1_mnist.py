from samples.tensorflow.docs.minist import input_data
import tensorflow as tf

"""
http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html
https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/examples/tutorials/mnist
"""
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([1, 10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1000):
        x_batch, y_batch = mnist.train.next_batch(100)
        train_step.run({x: x_batch, y_: y_batch})

    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
