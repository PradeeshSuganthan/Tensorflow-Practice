import tensorflow as tf
import time
import numpy as np 
import pandas as pd

#definitions
learning_rate = 0.1
batch_size = 1
n_epochs = 25
n_samples = 462
n_trainsamples = n_samples - n_samples/6
n_testsamples = n_samples/6
splitpoint = n_trainsamples
n_trainbatches = int(n_trainsamples/batch_size)
n_testbatches = int(n_testsamples/batch_size)

#read in data (HAVE TO FIGURE OUT HOW TO READ ALL DATA IN INSTEAD OF JUST FIRST ROW)
heart = ["tf-stanford-tutorials-master/data/heart.txt"]
# heart_queue = tf.train.string_input_producer(heart)

# reader = tf.TextLineReader()
# key, value = reader.read(heart_queue)
# record_defaults = [[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]]
# col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = tf.decode_csv(value, record_defaults=record_defaults)

# if col5 == "Present":
#  	col5 = "1"
# else:
# 	col5 = "0"

features = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9])
labels = tf.stack([col10])

#features_train, features_test = features[:splitpoint], features[splitpoint:]
#col10_train, col10_test = col10[:splitpoint], col10[splitpoint:]

print(col1)
print(col2)
print(col3)
print(features)
print(labels)
#create placeholders
X= tf.placeholder(tf.float32, [batch_size, 9], name="observed_vars") #input is each row, 9 classes
Y= tf.placeholder(tf.float32, [batch_size, 1], name="chd") #binary label


#weight and bias
W = tf.Variable(tf.random_normal([9, 1], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([1,]), name="bias")


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

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord)

	for i in range(n_epochs):
		for _ in range(n_trainbatches):
			X_batch = features
			Y_batch = labels
		#	X_batch = tf.train.batch([features], batch_size=batch_size)
		#	Y_batch = tf.train.batch([col10], batch_size=batch_size)

			print(X_batch.eval())
			print(X)
			print(Y_batch)
			print(Y)
			#example, label = sess.run([X_batch.eval(), Y_batch.eval()])
		#	print(example)
		#	print(label)
			sess.run([optimizer, loss], feed_dict = {X: X_batch.eval(), Y: Y_batch.eval()})

			print("Currently in train epoch " + str(i))



	print("Training complete")
#run test
	# total_correct_preds = 0
	# for i in range(n_testbatches):
	# 		X_batch, Y_batch = tf.train.batch([features, labels], batch_size=batch_size)
	# 		#Y_batch = tf.train.batch(col10_test, batch_size)
	# 		_, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict = {X: X_batch, Y: Y_batch})

	# 		preds = tf.nn.softmax(logits_batch)
	# 		correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
	# 		accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

	# 		total_correct_preds += sess.run(accuracy)

	# 		print("Currently in test epoch " + str(i))

	# coord.request_stop()
	# coord.join(threads)

	# print "Accuracy {0}".format(total_correct_preds/n_testsamples)
