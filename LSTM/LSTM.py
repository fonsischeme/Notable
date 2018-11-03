#!/home/athtrch/tensorflow_env/bin/python
"""
LSTM.py
--------------
A LSTM based network for classification.
Uses an character embedding layer, followed by a biLSTM.
"""

import numpy as np
import tensorflow as tf

class LSTM_Network(object):

    def stackedRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units, num_layers):
        n_hidden=hidden_units
        n_layers=num_layers
        # Prepare data shape to match `static_rnn` function requirements
        x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
        # print(x)
        # Define lstm cells with tensorflow
        # Forward direction cell

        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

            outputs, _ = tf.nn.static_rnn(lstm_fw_cell_m, x, dtype=tf.float32)
        return outputs[-1]

    def __init__(
        self, sequence_length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size, num_layers, num_classes):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length[0]], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length[1]], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.W = tf.placeholder(tf.float32, [vocab_size, embedding_size], name="W")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")

        # Embedding layer
        with tf.name_scope("embedding"):
            #self.W = tf.Variable(tf.constant(trainableEmbeddings, dtype=tf.float32),
            #    trainable=False, name="W")
            self.embedded_words1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            self.embedded_words2 = tf.nn.embedding_lookup(self.W, self.input_x2)

        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            self.out1=self.stackedRNN(self.embedded_words1, self.dropout_keep_prob, "side1", embedding_size, sequence_length[0], hidden_units, num_layers)
            self.out2=self.stackedRNN(self.embedded_words2, self.dropout_keep_prob, "side2", embedding_size, sequence_length[1], hidden_units, num_layers)

            combined = tf.concat((self.out1, self.out2), axis=1)
            combined_drop = tf.nn.dropout(combined, self.dropout_keep_prob)
            num_hidden_total = hidden_units * 2

            W = tf.get_variable(
                "W_o",
                shape=[num_hidden_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_o")

            self.logits = tf.nn.xw_plus_b(combined_drop, W, b, name="scores")

        with tf.name_scope("loss"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            self.loss_avg = tf.reduce_mean(self.loss)

        #### Accuracy computation is outside of this class.
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.round(tf.nn.sigmoid(self.logits)), tf.round(self.input_y))
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
