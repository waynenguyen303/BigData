import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
import os
from PIL import Image
from array import *
from random import shuffle
from sklearn import datasets, svm, metrics
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import datasets, linear_model
import numpy as np
from struct import unpack
import seaborn as sns
import matplotlib.pyplot as plt
import time


sess =tf.Session()

# set tensor variables -- given in class
tf.logging.set_verbosity(tf.logging.INFO)

x = tf.placeholder(tf.float32, [None, 4096],name='x')
W = tf.Variable(tf.zeros([4096, 5]),name='W')
b = tf.Variable(tf.zeros([5]),name='b')

y = tf.nn.softmax(tf.matmul(x, W) + b,name='y')
y_ = tf.placeholder(tf.float32, [None, 5],name='y_')
tf.add_to_collection('variables',W)
tf.add_to_collection('variables',b)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# save summaries for visualization
tf.summary.histogram('weights', W)
tf.summary.histogram('max_weight', tf.reduce_max(W))
tf.summary.histogram('bias', b)
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.histogram('cross_hist', cross_entropy)

# merge all summaries into one op
merged=tf.summary.merge_all()

#trainwriter=tf.summary.FileWriter('data/mnist_model'+'/logs/train',sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

fileimages = []
labels =[]
trainpath = "apparel/train"

folder = [x for x in os.listdir(trainpath) if os.path.isdir(os.path.join(trainpath,x))]

for item in folder:
    fileimages = fileimages + [trainpath + "/" + item + "/" + x for x in os.listdir(trainpath + "/" + item) if os.path.isfile(os.path.join(trainpath + "/" + item,x))]
    labels = labels + [item for x in os.listdir(trainpath + "/" + item) if os.path.isfile(os.path.join(trainpath + "/" + item,x))]
print(fileimages)
print(labels)


# step 1: load lists
filenames = tf.constant(fileimages)
labels1 = tf.constant(labels)

# step 2: create dataset returning slices
dataset = tf.data.Dataset.from_tensor_slices((filenames,labels1))

# step 3: parse every image in the dataset
def _parse_function(filenames, labels1):
    image_string = tf.read_file(filenames)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    resized_image = tf.image.resize_images(image, [64, 64])
    return resized_image, labels1

dataset = dataset.map(_parse_function)

#dataset = dataset.batch(500)


#step 4: create iterator and final input tensor
iterator = dataset.make_one_shot_iterator()

#images, labels1 = iterator.get_next()

summary_writer = tf.summary.FileWriter('./tensorflow/logdir', sess.graph)

# iterate thru each tensor --- can not get summary_writer to work to see on tensorboard
for i in range(488):

    im , lb = iterator.get_next()
    summ , _ = sess.run([im,lb])






