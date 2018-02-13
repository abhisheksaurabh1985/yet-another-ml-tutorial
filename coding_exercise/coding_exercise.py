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


# Get test train split
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(processed_data[:, 0:61],
                                                    processed_data[:, 61:63],
                                                    test_size=0.3,
                                                    random_state=42)

##########################
# CLASSIFICATION
#########################

# # CLASSIFICATION: LOGISTIC REGRESSION
# from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import train_test_split
# from sklearn import metrics
# from sklearn.cross_validation import cross_val_score
#
# # evaluate the model using 10-fold cross-validation. Assign a weight of 1 to false classification of low risk examples.
# # Assign a weight of 5 to false classification of high risk examples.
# lr_scores = cross_val_score(LogisticRegression(class_weight={1:1, 2:5}),
#                             processed_data[:, 0:60],
#                             processed_data[:,-1],
#                             scoring='accuracy',
#                             cv=10)
# print "Logistic regression cross validation scores", lr_scores
# print "Mean accuracy using Logistic regression after 10-CV:", lr_scores.mean()


# CLASSIFICATION: SVM

# Results: DNN
# ('Testing Accuracy:', 0.77666664)

"""
    labels = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    print labels.get_shape
    good, bad = tf.unstack(labels,  axis=1)
    loss_op = false_neg_cost*tf.reduce_sum(good) + \
                    false_pos_cost*tf.reduce_sum(bad)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(costwise_loss)

"""