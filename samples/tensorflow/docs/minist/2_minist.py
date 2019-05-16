from samples.tensorflow.docs.minist import input_data
import tensorflow as tf

"""
http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html
"""


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='W')


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='b')


def conv_2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 10])

# 第一层卷积
with tf.variable_scope('conv1'):
    W = weight_variable([5, 5, 1, 32])
    b = bias_variable([32])
    h_conv1 = tf.nn.relu(conv_2d(x_image, W) + b)
    h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
with tf.variable_scope('conv2'):
    W = weight_variable([5, 5, 32, 64])
    b = bias_variable([64])
    h_conv2 = tf.nn.relu(conv_2d(h_pool1, W) + b)
    h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层
with tf.variable_scope('fc1'):
    W = weight_variable([7 * 7 * 64, 1024])
    b = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W) + b)

# Dropout
with tf.variable_scope('dropout'):
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
with tf.variable_scope('fc2'):
    W = weight_variable([1024, 10])
    b = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W) + b)

cross_entropy = -tf.reduce_mean(y_ * tf.log(y_conv))
# 注意可以对比学习率0.1与1e-4在训练起来的区别！
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()

# 输出可训练参数
# https://blog.csdn.net/Jerr__y/article/details/70809528
tvs = tf.trainable_variables()
print('There are %d train_able_variables in the Graph: ' % len(tvs))
for v in tvs:
    print(v)

with tf.Session() as sess:
    sess.run(init)

    for i in range(10000):
        x_batch, y_batch = mnist.train.next_batch(64)
        train_step.run({x: x_batch, y_: y_batch, keep_prob: 0.5})
        if i % 100 == 0:
            train_accuracy = accuracy.eval({x: x_batch, y_: y_batch, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

    print("test accuracy %g" % accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
