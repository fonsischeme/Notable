"""
train.py
--------
This script trains an ae Neural Network Model.

Input
-----
csv_file File with data

Output
------
Checkpoint Files and Performance Summary Files
"""
import datetime
import os
import sys
import time

import argparse
import gensim
import json
import numpy as np
from sklearn.model_selection import train_test_split
import string
import tensorflow as tf

from LSTM import LSTM_Network
import data_utils

def main(args):
    max_length = list(map(int, args.max_length.split(",")))
    x1, x2, y, trainableEmbeddings, labels_map = data_utils.data_loader(args.train_file, args.embeddings, max_length)

    train_x1, dev_x1, train_x2, dev_x2, train_y, dev_y = train_test_split(x1, x2, y, test_size=0.2)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=args.allow_soft_placement,
          log_device_placement=args.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            LSTM = LSTM_Network(sequence_length = max_length,
                            vocab_size=trainableEmbeddings.shape[0],
                            embedding_size=trainableEmbeddings.shape[1],
                            hidden_units=args.num_hidden,
                            l2_reg_lambda=args.l2_reg_lambda,
                            batch_size=args.batch_size,
                            num_layers=args.num_layers,
                            num_classes=args.num_classes
                        )

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(args.learning_rate)
            grads_and_vars = optimizer.compute_gradients(LSTM.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(args.output, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Acc and Loss Summaries
            loss_summary = tf.summary.scalar("loss", LSTM.loss_avg)
            acc_summary = tf.summary.scalar("accuracy", LSTM.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph_def)

            # Checkpointing
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            # Tensorflow assumes this directory already exists so we need to create it
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=args.num_checkpoints)

            sess.run(tf.initialize_all_variables())

            def train_step(x_batch):
                """
                A single training step
                """
                feed_dict = {
                  LSTM.input_x1: x_batch[0],
                  LSTM.input_x2: x_batch[1],
                  LSTM.input_y: x_batch[2],
                  LSTM.dropout_keep_prob: args.dropout_keep_prob,
                  LSTM.W: trainableEmbeddings
                }
                _, step, summaries, loss, acc = sess.run(
                    [train_op, global_step, train_summary_op, LSTM.loss, LSTM.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {}, acc {}".format(time_str, step, loss, acc))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  LSTM.input_x1: x_batch[0],
                  LSTM.input_x2: x_batch[1],
                  LSTM.input_y: x_batch[2],
                  LSTM.dropout_keep_prob: 1.0,
                  LSTM.W: trainableEmbeddings
                }
                step, summaries, loss, acc  = sess.run(
                    [global_step, dev_summary_op, LSTM.loss, LSTM.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {}, acc {}".format(time_str, step, loss, acc))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_utils.iterator(list(zip(train_x1,train_x2,train_y)), args.batch_size, args.num_epochs)

            # Training loop. For each batch...
            for batch in batches:
                x_batch, dev = batch
                x_batch = zip(*x_batch)

                if dev:
                    print("\nEvaluation:")
                    dev_step((dev_x1, dev_x2, dev_y), writer=dev_summary_writer)
                    print("")

                train_step(x_batch)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % args.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            print("\nEvaluation:")
            dev_step((dev_x1, dev_x2, dev_y), writer=dev_summary_writer)
            print("")

    print("COMPLETE")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("train_file", action='store', help="Text Data")
    parser.add_argument("embeddings", action='store', help="Doc2Vec Model File")
    parser.add_argument("-a", "--allow_soft_placement", action="store_false",
                        help="Allow device soft device placement")
    parser.add_argument("-b", "--batch_size", action="store", type=int,
                        default=64, help="Batch Size (default: 1000)")
    parser.add_argument("-c", "--num_classes", action="store", type=int,
                        default=34, help="Number of Classes (default: 34)")
    parser.add_argument("-ce", "--checkpoint_every", action="store", type=int,
                        default=1000,
                        help="Save model after this many steps (default: 100)")
    parser.add_argument("-dp", "--dropout_keep_prob", action="store", type=float,
                        default=0.5, help="Dropout keep probability (default: 0.5)")
    parser.add_argument("-e", "--num_epochs", action="store", type=int,
                        default=300,
                        help="Number of times the training data is seen by the model (default: 30)")
    parser.add_argument("-m", "--max_length", action="store", type=str,
                        default='15,50',
                        help="Maximun Length of Reviews (default: 20)")
    parser.add_argument("-l", "--learning_rate", action="store", type=float,
                        default=0.0001,
                        help="The learning rate for the optimizer")
    parser.add_argument("-ld", "--log_device_placement", action="store_true",
                        help="Log placement of ops on devices")
    parser.add_argument("-nc", "--num_checkpoints", action="store", type=int,
                        default=5,
                        help="Number of checkpoints to store (default: 5)")
    parser.add_argument("-nh", "--num_hidden", action="store", type=int,
                        default=50,
                        help="Number of neurons in Hidden Layers (default: 500)")
    parser.add_argument("-nl", "--num_layers", action="store", type=int,
                        default=3, help="Number of LSTM Layers (default: 3)")
    parser.add_argument("-r", "--l2_reg_lambda", action="store", type=float,
                        default=0.0,
                        help="The learning rate for the optimizer")
    parser.add_argument("-o", "--output", action="store", type=str,
                        help="Output Directory", default='outputs')
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose is turned one!")
    argv = parser.parse_args()
    sys.exit(main(args=argv))
