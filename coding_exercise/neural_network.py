#! /usr/bin/python
# Train a neural network to predict on the german credit data
import time
import signal
import sys

import tensorflow as tf
import numpy as np
from numpy import genfromtxt

from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

def random_batch(dataset, batch_size):
    sample = dataset[np.random.choice(dataset.shape[0], batch_size, replace=False),:]
    last_col_index = dataset.shape[1]-2
    x = sample[:,0:last_col_index]
    y = sample[:,last_col_index:last_col_index+2]
    return (x, y)

# Tensorflow convinience functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Constants
num_classes = 2

learning_rate = 1e-4
num_neurons = 124
batch_size = 100
num_k_folds = 10
dropout = 0.5

false_neg_cost = 1.0
false_pos_cost = 5.0

# Read the entire dataset (assumed to have already been preprocessed)
# dataset = genfromtxt('newData.csv', delimiter=',')
# rows, cols = dataset.shape

import os
import numpy as np
import pandas as pd

from sklearn import preprocessing

#########################
# DATA PRE-PROCESSING
########################

os.chdir('/home/abhishek/Desktop/Projects/tf/yet_another_ML_tutorial/coding_exercise/')
raw_data = pd.read_csv("./CreditDataset.csv", header=None)
print "Shape of original data frame:", raw_data.shape

# Get data types
print raw_data.dtypes
obj_df = raw_data.select_dtypes(include=['object']).copy()
print "Shape of object data frame:", obj_df.shape
int_df = raw_data.select_dtypes(include=['int64']).copy()
print "Shape of int64 data frame:", int_df.shape
print "Type of int data frame:", type(int_df)

# Check for null values in the columns containing categorical variables
print obj_df[obj_df.isnull().any(axis=1)]


# One hot encoding of the columns containing categorical variables
# Label encoder
# 1. INSTANTIATE
# encode labels with value between 0 and n_classes-1.
le = preprocessing.LabelEncoder()
# FIT AND TRANSFORM. use df.apply() to apply le.fit_transform to all columns
le_obj_df = obj_df.apply(le.fit_transform)
# print raw_data.select_dtypes(include=['object']).head(5)
# print le_obj_df.head()

# One hot encoding of categorical variables
# 1. INSTANTIATE
encode_object = preprocessing.OneHotEncoder()
# 2. FIT
encode_object.fit(le_obj_df)
# 3. Transform
onehotlabels = encode_object.transform(le_obj_df).toarray()
print onehotlabels.shape
print type(onehotlabels)

# Merge the int64 data frame with the one hot labels
np_int_df = int_df.as_matrix()
print np_int_df.shape
processed_data = np.concatenate([onehotlabels, np_int_df], axis=1)
print processed_data.shape

# print processed_data[:,-1]
print processed_data.dtype


# Encode output for NN
# encode_output = preprocessing.OneHotEncoder()
# encode_output.fit(processed_data[:,-1])
# output_onehotlabels = encode_output.transform(processed_data[:,-1])
# print "Shape of encoded one hot labels:", output_onehotlabels.shape
#

raw_labels = np.array(processed_data[:,-1]).astype(int)
encoded_labels = np.zeros((processed_data[:,-1].shape[0], 2))
encoded_labels[np.arange(processed_data[:,-1].shape[0]), raw_labels-1] = 1


processed_data = processed_data[:,0:61]
processed_data = np.concatenate([processed_data, encoded_labels], axis=1)
print processed_data.shape


dataset = processed_data
dataset.astype(dtype=float)
rows, cols = dataset.shape


x_width = cols-2
# assume the last 2 columns are the label

# Placeholder values
x = tf.placeholder(tf.float32, [None, x_width])

# Neural network with 2 hidden layers

# Fully connected layer 1:
w_fc1 = weight_variable([x_width, num_neurons])       # weights
b_fc1 = bias_variable([num_neurons])                  # biases
h_fc1 = tf.nn.relu(tf.matmul(x, w_fc1) + b_fc1)       # activation
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)          # dropout

# Fully connected layer 2:
w_fc2 = weight_variable([num_neurons, num_neurons])
b_fc2 = bias_variable([num_neurons])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# Fully connected layer 3:
w_fc3 = weight_variable([num_neurons, num_neurons])
b_fc3 = bias_variable([num_neurons])
h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, w_fc3) + b_fc3)
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

# Readout layer
w_fc_out = weight_variable([num_neurons, num_classes])
b_fc_out = bias_variable([num_classes])

# The softmax function will make probabilties of Good vs Bad score at the output
y_ = tf.nn.softmax(tf.matmul(h_fc3_drop, w_fc_out) + b_fc_out)
y = tf.placeholder(tf.float32, [None, num_classes])

# Training
# Different loss functions:
# cross_entropy =  -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-15, 1.0)))

# Mofified mean square error function which uses a cost of 5 for false positives
# and a cost of one for false negatives
square_diff = tf.square(y - y_)
# good, bad = tf.split(1, 2, square_diff)
good, bad = tf.unstack(square_diff,  axis=1)
costwise_loss = false_neg_cost*tf.reduce_sum(good) + false_pos_cost*tf.reduce_sum(bad)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(costwise_loss)

init = tf.initialize_all_variables()

# Create session
sess = tf.InteractiveSession()
sess.run(init)

# Use 10-Fold cross validation to find the avg validation accuracy and
# confusion matrix values
kf = KFold(rows, n_folds=num_k_folds)
fold_counter = 1
val_conmats = []
val_precisions = []
val_recalls = []
val_f_scores = []

for train_indices, val_indices in kf:
    # split the data into train and validation
    train_dataset = dataset[train_indices,:]
    val_dataset = dataset[val_indices,:]

    for i in range(3000):
      batch_xs, batch_ys = random_batch(train_dataset, batch_size)
      sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})

    # Get the ground truth values for this k-fold
    yT = np.argmax(val_dataset[:, 61:63], axis=1)

    # Evaluate the predicted values
    y_p = tf.argmax(y_,1)
    yP = sess.run(y_p, feed_dict={
         x: val_dataset[:, 0:61], y: val_dataset[:, 61:63], keep_prob: 1.0})
    print("Fold #: %d, validation accuracy: " % (fold_counter))

    # Metrics and confusion matrix
    [precision, recall, f_score, _] = precision_recall_fscore_support(yT, yP, average='macro')
    print("Validation k-fold #%d - precision: %f, recallL: %f, f-score: %f" % (
           fold_counter, precision, recall, f_score))
    conmat = confusion_matrix(yT, yP)
    print("Confusion matrix:")
    print("Good | Bad Credit")
    print conmat

    val_precisions.append(precision)
    val_recalls.append(recall)
    val_f_scores.append(f_score)
    val_conmats.append(conmat)
    fold_counter = fold_counter + 1

print("\nAveraging the 10-fold results:")
print("Validation precision - mean: %f, stddev: %f" % (
       np.mean(val_precisions), np.std(val_precisions)))
print("Validation recall - mean: %f, stddev: %f" % (
       np.mean(val_recalls), np.std(val_recalls)))
print("Validation f-score - mean: %f, stddev: %f" % (
       np.mean(val_f_scores), np.std(val_f_scores)))
print("Confusion matrix:")
print("Good | Bad Credit")
print (sum(val_conmats)).astype(float) / num_k_folds