import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import numpy as np 

#definitions
learning_rate = 0.1
batch_size = 128
n_epochs = 25


#read in data
MNIST = input_data.read_data_sets("MNIST_data", one_hot = True)


#create placeholders
X= tf.placeholder(tf.float32, [batch_size, 784], name="image")
Y= tf.placeholder(tf.float32, [batch_size, 10], name="label")


#weight and bias
W = tf.Variable(tf.random_normal([784, 10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([1,10]), name="bias")


#construct model
logits = tf.matmul(X, W) + b



#create loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
loss = tf.reduce_mean(entropy)


#loss minimization optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)



init = tf.global_variables_initializer()

#run training

with tf.Session() as sess:
	sess.run(init)
	n_batches = int(MNIST.train.num_examples/batch_size)
	for i in range(n_epochs):
		for _ in range(n_batches):
			X_batch, Y_batch, = MNIST.train.next_batch(batch_size)
			sess.run([optimizer, loss], feed_dict = {X: X_batch, Y: Y_batch})



#run test
	n_batches = int(MNIST.test.num_examples/batch_size)
	total_correct_preds = 0
	for i in range(n_batches):
			X_batch, Y_batch, = MNIST.test.next_batch(batch_size)
			_, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict = {X: X_batch, Y: Y_batch})

			preds = tf.nn.softmax(logits_batch)
			correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
			accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

			total_correct_preds += sess.run(accuracy)


	print "Accuracy {0}".format(total_correct_preds/MNIST.test.num_examples)
