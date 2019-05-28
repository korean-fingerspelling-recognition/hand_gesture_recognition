import tensorflow as tf
import util
import layers
import numpy as np
import os

# flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("append_handmask", False, 'If true, train with handmask appended')
flags.DEFINE_integer("image_size", 64, "define input image size")
flags.DEFINE_float("dropout", 0.5, "dropout rate")
flags.DEFINE_integer("display_step", 1, "define display_step")
flags.DEFINE_boolean("use_bottleneck", False, "change the sub-layer on Resnet")
flags.DEFINE_string("saved_path", "Saved", "path to save model: Will create automatically if it does not exist")
flags.DEFINE_integer("num_steps", 555555500, "define the number of steps to train")
flags.DEFINE_boolean("train_mode", True, "True if in train mode, False if in Test mode")
flags.DEFINE_string("data_path", "./new_dataset", "path of the dataset: inside this directory there should be files named with label")
flags.DEFINE_integer("learning_rate", 0.0001, "learning_rate during training")
flags.DEFINE_integer("batch_size", 64, "batch size during training")
flags.DEFINE_integer("save_steps", 100, "number of save steps")
flags.DEFINE_integer("dim", 3, "input_channel is 4")
flags.DEFINE_integer("num_classes", 7, "number of classes")
flags.DEFINE_integer("model_num", 0, "0: default model / 1: resnet / 2: bottlenect resnet")

# Input Graph
X = tf.placeholder(tf.float32,[None,FLAGS.image_size,FLAGS.image_size,FLAGS.dim], name="Input")
Y = tf.placeholder(tf.uint8,[None,FLAGS.num_classes], name="Target")

# Tensorboard stuff
# global_step = tf.train.get_or_create_global_step()

weights = {
    'wc1': tf.Variable(tf.random_normal([5,5,FLAGS.dim,32])),
    'wc2': tf.Variable(tf.random_normal([5,5,32,64])),
    'wd1': tf.Variable(tf.random_normal([int(FLAGS.image_size*FLAGS.image_size/16)*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024,FLAGS.num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([FLAGS.num_classes]))
}

# Creating the model
if (FLAGS.model_num == 0):
	print("Training by Original Model")
	logits = layers.convnet(X,weights,biases,FLAGS.dropout)
else: 
	logits = layers.resnet(X)
prediction = tf.nn.softmax(logits,name='prediction')

# Defining the loss
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y), name="Loss")
optimiser = tf.train.AdamOptimizer(FLAGS.learning_rate,name='optimiser')
training_op =  optimiser.minimize(loss_op,name='training')

# Evaluating Model
correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1),name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32),name='accuracy')

tf.summary.scalar(name="Loss", tensor=loss_op)
tf.summary.scalar(name="Accuracy", tensor=accuracy)
tf.summary.image(name="Input_images", tensor=X)
summary = tf.summary.merge_all()

saver = tf.train.Saver()

# Initializer
init = tf.global_variables_initializer()
saver_step = FLAGS.save_steps
with tf.Session() as sess:
    dir = os.path.join('./tensorboard', FLAGS.saved_path)
    if not os.path.exists(dir): os.makedirs(dir)
    writer = tf.summary.FileWriter(logdir=dir, graph=sess.graph)

    sess.run(init)

    for step in range(1,FLAGS.num_steps+1):

        # Get the next batch
        batch_x, batch_y = util.get_next_batch(batch_size=FLAGS.batch_size, image_size=FLAGS.image_size)
	batch_y = tf.one_hot(np.reshape(batch_y, [-1]),FLAGS.num_classes)

	_, s = sess.run([training_op, summary],feed_dict={X:batch_x,Y:sess.run(batch_y)})

        writer.add_summary(s, step)

        if step%FLAGS.display_step == 0 or step == 1:
            loss,acc = sess.run([loss_op,accuracy],feed_dict={X:batch_x,Y:sess.run(batch_y)})
            print("Step: {} / Loss: {}  / Accuracy: {}".format(step, loss, acc))
        if step%saver_step == 0:
	     if (not os.path.exists(FLAGS.saved_path)): 
		os.mkdir(FLAGS.saved_path)
		saver.save(sess,save_path=os.path.join(FLAGS.saved_path, str(step+1)))

    print('Done Training!!')
