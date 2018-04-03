#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 22:41:27 2018

@author: lizhaosheng
"""

#最简单的神经网络，两层
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#download data    one-hot: label is 0 0 0 1 0 0 
mnist = input_data.read_data_sets("MNIST_data",one_hot = True)

#size of each batch
batch_size = 100
#calculate how many batches     //divide
n_batch = mnist.train.num_examples//batch_size

#define two placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#create a simple neural network
W = tf.Variable(tf.ones([784,10]))
b = tf.Variable(tf.ones([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#quadratic cost function
loss = tf.reduce_mean(tf.square(y-prediction))

#gradient descent
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#init
init = tf.global_variables_initializer()

#restore in an bool list
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

#get the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict ={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict = {x:mnist.test.images,y:mnist.test.labels})
        print ("Iter: " + str(epoch)) 
        print ("Testing Accuracy:" + str(acc))
