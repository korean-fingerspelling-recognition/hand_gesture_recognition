import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def conv2d(inputs, weights, biases, strides=1):
    inputs = tf.nn.conv2d(inputs, weights, [1,1,1,1],padding = 'SAME')
    inputs = tf.nn.bias_add(inputs,biases)
    return tf.nn.relu(inputs)

def maxpool2d(inputs,k=2):
    return tf.nn.max_pool(inputs, [1,k,k,1],[1,k,k,1],padding='SAME')

def convnet(inputs, weights, biases, keep_prob):
    # Layer 1
    conv1 = conv2d(inputs,weights['wc1'],biases['bc1'])
    conv1 = maxpool2d(conv1)

    # Layer 2
    conv2 = conv2d(conv1,weights['wc2'],biases['bc2'])
    conv2 = maxpool2d(conv2)

    # Fully Connected Layer
    fc1 = tf.reshape(conv2, [-1,weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1,keep_prob)

    # Output
    return tf.add(tf.matmul(fc1,weights['out']),biases['out'],name='logits')

def resnet(inputs):
	with tf.variable_scope("network", reuse=FLAGS.train_mode):
		if FLAGS.model_num==2: # in case of deep model
			residual_block = bottle_resblock
		else: #FLAGS.model_num==1
			residual_block = resblock

		ch = 32
		x = conv(inputs, ch, 3, 1, "conv")
		for i in range(2):
			print("channel number 64: {}".format(i))
			x = residual_block(x, ch*2, FLAGS.train_mode, True, True) # ch = 128
		x = residual_block(x, ch*4, FLAGS.train_mode, True, False)
		for i in range(2):
			x = residual_block(x, ch*4, FLAGS.train_mode, True, True) # ch = 256
		x = residual_block(x, ch*8, FLAGS.train_mode, True, False)
		x = batch_norm(x, FLAGS.train_mode, ch*8, scope = 'batch_norm')
		x = tf.nn.relu(x)
		# x = global_avg_pooling(x)
		x = fully_connected(x, units=FLAGS.num_classes, scope='logit')
		return x

'''a single block for ResNet v2, without a bottleneck'''
def resblock(x_init, channels, is_training=True, use_bias=True, 
		downsample=False, scope='resblock') :
    with tf.variable_scope(scope) :
 	f_channel = x_init.get_shape().as_list()[-1]
        x = batch_norm(x_init, is_training, f_channel, scope='batch_0')
        x = tf.nn.relu(x)
        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            x_init = conv(x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')
        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')
        x = batch_norm(x, is_training, channels, scope='batch_1')
        x = tf.nn.relu(x)
        x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')
        return x + x_init

''' in case of making a model very deep '''
def bottle_resblock(x_init, channels, is_training=True, use_bias=True, 
							downsample=False, scope='bottle_resblock') :
    with tf.variable_scope(scope) :
        x = batch_norm(x_init, is_training, channels, scope='batch_2')
        shortcut = relu(x)

        x = conv(shortcut, channels, kernel=1, stride=1, use_bias=use_bias, 
						scope='conv_1x1_front')
        x = batch_norm(x, is_training, channels, scope='batch_3')
        x = relu(x)

        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, 
							scope='conv_0')
            shortcut = conv(shortcut, channels*4, kernel=1, stride=2, 
									use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, 
							scope='conv_0')
            shortcut = conv(shortcut, channels * 4, kernel=1, stride=1, 
									use_bias=use_bias, scope='conv_init')

        x = batch_norm(x, is_training, channels, scope='batch_4')
        x = relu(x)
        x = conv(x, channels*4, kernel=1, stride=1, use_bias=use_bias, 
						scope='conv_1x1_back')

        return x + shortcut

def conv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             strides=stride, use_bias=use_bias, padding="SAME")
	return x

def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, 
									kernel_regularizer=tf.contrib.layers.xavier_initializer(), use_bias=use_bias)
        return x


def batch_norm(input_layer, is_training, dimension, scope = 'batch_norm'):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    with tf.variable_scope(scope):
    	mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    	beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    	gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    	bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, 0.001)

    	return bn_layer
