import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        """

        :param sequence_length: Length of sentences. All the sentences will have the same length (59 in this case).
        :param num_classes: Number of classes in the output layer. Two in this case (positive and negative).
        :param vocab_size: Size of vocabulary. Needed to define the size of embedding layer which will have the shape
        (vocabulary_size, embedding_size).
        :param embedding_size: Dimensionality of embeddings.
        :param filter_sizes: Number of words we want our convolution filter to cover. We have num_filters for each size
        specified here. For e.g., [3,4,5] means that we will have filters that slide over 3,4 and 5 words respectively,
        for a total of 3*num_filters filter.
        :param num_filters: Number of filters per filter size.
        :param l2_reg_lambda:
        """

        # Placeholders for input, output and dropout
        # tf.placeholder creates a placeholder variable that we feed to the network when we execute it at train or test
        # time. The second argument is the shape of the input tensor. None means that the length of that dimension could
        # be anything. In our case, the first dimension is the batch size, and using None allows the network to handle
        # arbitrarily sized batches.
        # Dropout: The probability of keeping a neuron in the dropout layer is also an input to the network because we
        # enable dropout only during training. We disable it when evaluating the model (more on that later).
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer: Essentially a look up table that will be learnt from the data. Maps vocabulary word indices
        # into a low dimensional vector representation.
        # tf.device("/cpu:0") forces an operation to be executed on the CPU. By default TensorFlow will try to put the
        # operation on the GPU if one is available, but the embedding implementation doesn't currently have GPU support
        # and throws an error if placed on the GPU.
        # tf.name_scope() creates a new Name Scope with the name 'embedding'. The scope adds all operations into a top-
        # level node called 'embedding' so that you get a nice hierarchy when visualizing your network in TensorBoard.
        # W is our embedding matrix that we learn during training. We initialize it using a random uniform distribution.
        # tf.nn.embedding_lookup creates the actual embedding operation. The result of the embedding operation is a
        # 3-dimensional tensor of shape [None, sequence_length, embedding_size].
        # TensorFlow's convolutional conv2d operation expects a 4-dimensional tensor with dimensions corresponding to
        # batch, width, height and channel. The result of our embedding doesn't contain the channel dimension, so we add
        # it manually, leaving us with a layer of shape [None, sequence_length, embedding_size, 1].
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size],
                                                   -1.0, 1.0),
                                 name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        # Using the feature vector from max-pooling (with dropout applied) we can generate predictions by doing a matrix
        # multiplication and picking the class with the highest score. We could also apply a softmax function to convert
        # raw scores into normalized probabilities, but that wouldn't change our final predictions.
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        # Here, tf.nn.softmax_cross_entropy_with_logits is a convenience function that calculates the cross-entropy loss
        # for each class, given our scores and the correct input labels. We then take the mean of the losses. We could
        # also use the sum, but that makes it harder to compare the loss across different batch sizes and train/dev data
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")