import numpy as np
import tensorflow as tf
import six

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
	with tf.variable_scope("network", reuse=not FLAGS.train_mode):
		if FLAGS.model_num==2: # in case of deep model
			residual_block = bottle_resblock
		else: #FLAGS.model_num==1
			residual_block = resblock

		ch = 32
		x = conv(inputs, ch, 3, 1, "conv")
		for i in range(2):
			x = residual_block(x, ch*2, FLAGS.train_mode, True, True) # ch = 128
		x = residual_block(x, ch*4, FLAGS.train_mode, True, False)
		for i in range(2):
			x = residual_block(x, ch*4, FLAGS.train_mode, True, True) # ch = 256
		x = residual_block(x, ch*8, FLAGS.train_mode, True, False)
		x = batch_norm(x, FLAGS.train_mode, scope = 'batch_norm')
		x = tf.nn.relu(x)
		# x = global_avg_pooling(x)
		x = fully_connected(x, units=FLAGS.num_classes, scope='logit')
		return x

'''a single block for ResNet v2, without a bottleneck'''
def resblock(x_init, channels, is_training=True, use_bias=True, 
		downsample=False, scope='resblock') :
    with tf.variable_scope(scope) :
        x = batch_norm(x_init, is_training, scope='batch_norm_0')
        x = tf.nn.relu(x)
        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            x_init = conv(x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')
        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')

        x = batch_norm(x, is_training, scope='batch_norm_1')
        x = tf.nn.relu(x)
        x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')
        return x + x_init

''' in case of making a model very deep '''
def bottle_resblock(x_init, channels, is_training=True, use_bias=True, 
							downsample=False, scope='bottle_resblock') :
    with tf.variable_scope(scope) :
        x = batch_norm(x_init, is_training, scope='batch_norm_1x1_front')
        shortcut = relu(x)

        x = conv(shortcut, channels, kernel=1, stride=1, use_bias=use_bias, 
						scope='conv_1x1_front')
        x = batch_norm(x, is_training, scope='batch_norm_3x3')
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

        x = batch_norm(x, is_training, scope='batch_norm_1x1_back')
        x = relu(x)
        x = conv(x, channels*4, kernel=1, stride=1, use_bias=use_bias, 
						scope='conv_1x1_back')

        return x + shortcut

def conv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding)
			return x

def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, 
									kernel_regularizer=weight_regularizer, use_bias=use_bias)
        return x

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)
