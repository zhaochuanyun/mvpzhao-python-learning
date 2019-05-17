from samples.tensorflow.docs.minist import input_data
import tensorflow as tf


def weights_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name='b')


def conv_2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

with tf.variable_scope('conv1'):
    W_conv1 = weights_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv_2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1)

with tf.variable_scope('conv2'):
    W_conv2 = weights_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv_2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2)

with tf.variable_scope('fc1'):
    W_fc1 = weights_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

with tf.variable_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.variable_scope('fc2'):
    W_fc2 = weights_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y = tf.nn.softmax((tf.matmul(h_fc1_drop, W_fc2) + b_fc2))

cross_entropy = -tf.reduce_mean(y_ * tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

vbs = tf.trainable_variables()
print('There are %d train_able_variables in the graph.' % len(vbs))
for vb in vbs:
    print(vb)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        x_batch, y_batch = mnist.train.next_batch(64)
        train_step.run({x: x_batch, y_: y_batch, keep_prob: 0.5})
        train_accuracy = accuracy.eval({x: x_batch, y_: y_batch, keep_prob: 1.0})
        if i % 100 == 0:
            print('step %d, trainning accurucy: %g' % (i, train_accuracy))

    print("test accuracy %g" % accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
