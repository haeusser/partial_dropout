import tensorflow as tf
import random
from random import randint
from tensorflow.examples.tutorials.mnist import input_data

# Set Random-Seeds
random.seed(1234)
tf.set_random_seed(1234)

# Read Dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Parameters
input_dimension = 784
output_dimension = 10


x = tf.placeholder(tf.float32, shape=[None, input_dimension])
y_ = tf.placeholder(tf.float32, shape=[None, output_dimension])
#keep_prob = tf.placeholder(tf.float32)
keep_prob_fc1 = tf.placeholder(tf.float32)
keep_prob_fc2 = tf.placeholder(tf.float32)

sess = tf.InteractiveSession()


# Build the Model

# Build the Model

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride_size):
  return tf.nn.conv2d(x, W, strides=[1, stride_size, stride_size, 1], padding='SAME')

def max_pool_3x3(x, k_size, stride_size):
  return tf.nn.max_pool(x, ksize=[1, k_size, k_size, 1], strides=[1, stride_size, stride_size, 1], padding='VALID')

  
x_image = tf.reshape(x, [-1,28,28,1])  
  
# conv1, pool1  
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 1) + b_conv1)
h_pool1 = max_pool_3x3(h_conv1, 2, 2)


# conv2, pool2  
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 1) + b_conv2)
h_pool2 = max_pool_3x3(h_conv2, 2, 2)


# fc1_drop
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool5_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_fc1)


# fc2_drop
W_fc2 = weight_variable([1024, 1024])
b_fc2 = bias_variable([1024])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob_fc2)


# softm
W_softm =  weight_variable([1024, 10])
b_softm = bias_variable([10])

y_conv = tf.matmul(h_fc2_drop, W_softm) + b_softm

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))



#Train the model 

##train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

sum_train_accuracy = 0

for i in range(2000000):
    
    batch = mnist.train.next_batch(100)

    ##test the model
    if i%600 == 0:
        sum_test_accuracy = 0
        for j in range(100):
            batch_test = mnist.test.next_batch(100)
            #train_accuracy = accuracy.eval(feed_dict={x:batch_train[0], y_: batch_train[1], keep_prob_fc1: 1.0, keep_prob_fc2: 1.0})
            sum_test_accuracy += accuracy.eval(feed_dict={x:batch_test[0], y_: batch_test[1], keep_prob_fc1: 1.0, keep_prob_fc2: 1.0})
        overall_test_accuracy = (sum_test_accuracy/100.)
        print("step %d, test accuracy %f"%((i/600.), overall_test_accuracy))
	
    ##Define at which layer dropout is applied
    prob_fc1 = 1.0
    prob_fc2 = 1.0
    rand_num = random.randint(1, 2)
    if rand_num == 1:
        prob_fc1 = 0.5
    elif rand_num == 2:
        prob_fc2 = 0.5
    #print (prob_fc1)
    #print (prob_fc2)	
    
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob_fc1: prob_fc1, keep_prob_fc2: prob_fc2}) 

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob_fc1: 1.0, keep_prob_fc2: 1.0}))
