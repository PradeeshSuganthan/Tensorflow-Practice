import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import numpy as np 
import csv

#definitions
learning_rate = 0.1
batch_size = 462
n_epochs = 25
labels = np.zeros((462,2))


#read in data
heart = "tf-stanford-tutorials-master/data/heart.txt"

f = open(heart, 'r')
next(f)
reader = (row.strip().split() for row in f)
cols = list(zip(*reader))

col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = cols

col1 = list(col1)
col1 = map(int, col1)
col2 = list(col2)
col2 = map(float, col2)
col3 = list(col3)
col3 = map(float, col3)
col4 = list(col4)
col4 = map(float, col4)
col5 = list(col5)
col5 = map(int, col5)
col6 = list(col6)
col6 = map(int, col6)
col7 = list(col7)
col7 = map(float, col7)
col8 = list(col8)
col8 = map(float, col8)
col9 = list(col9)
col9 = map(int, col9)
col10 = list(col10)
col10 = map(int, col10)

#one-hot encode labels
targets = np.array(col10)
labels[np.arange(462), targets] = 1

features = np.column_stack([col1, col2, col3, col4, col5, col6, col7, col8, col9])

#create placeholders
X= tf.placeholder(tf.float32, [batch_size, 9], name="image")
Y= tf.placeholder(tf.float32, [batch_size, 2], name="label")


#weight and bias
W = tf.Variable(tf.random_normal([9, 2], stddev=0.01), name="weights")
#b = tf.Variable(tf.zeros([1,10]), name="bias")


#construct model
logits = tf.matmul(X, W) #+ b



#create loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
loss = tf.reduce_mean(entropy)


#loss minimization optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)



init = tf.global_variables_initializer()

#run training

with tf.Session() as sess:
	sess.run(init)
#	n_batches = int(MNIST.train.num_examples/batch_size)
	print "Training"
	X_batch = features
	Y_batch = labels
	for i in range(n_epochs):
#		for _ in range(n_batches):
#		X_batch, Y_batch, = MNIST.train.next_batch(batch_size)
		sess.run([optimizer, loss], feed_dict = {X: X_batch, Y: Y_batch})
		print "Epoch " + str(i + 1) + " complete"


#run test
	print "Testing"
#	n_batches = int(MNIST.test.num_examples/batch_size)
	total_correct_preds = 0
#			X_batch, Y_batch, = MNIST.test.next_batch(batch_size)
	_, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict = {X: X_batch, Y: Y_batch})

	preds = tf.nn.softmax(logits_batch)
	correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
	accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

	total_correct_preds += sess.run(accuracy)


	print "Accuracy: {0}".format((total_correct_preds/batch_size)*100) + " %"
