from comet_ml import Experiment
# import numpy as np
from functools import partial
import tensorflow as tf
# from datetime import datetime

experiment = Experiment(project_name="sweep-langmuir-ml", workspace="physicistphil")

def build_graph(hyperparams):
    n_inputs = 500
    scale = 0.01
    # TODO: implmenet learning rate schedule
    learning_rate = 1e-3
    momentum = 0.9
    # TODO: implement dropout
    # dropout_rate = 0.5

    dense_layer = partial(tf.layers.dense, actiavtion=tf.nn.relu,
                          kernel_initializer=tf.contrib.layers.xavier_intializer(),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale))

    with tf.name_scope("data"):
        X = tf.placeholder(tf.float32, [None, n_inputs], name="X")

    with tf.name_scope("nn"):
        enc1 = dense_layer(X, 200, name="enc1")
        enc2 = dense_layer(enc1, 100, name="enc2")
        enc3 = dense_layer(enc2, 50, name="enc3")
        h_base = dense_layer(enc3, 20, name="h_base")
        dec1 = dense_layer(h_base, 50, name="dec1")  # TODO: maybe implement weight tying
        dec2 = dense_layer(dec1, 100, name="dec2")
        dec3 = dense_layer(dec2, 200, name="dec4")
        output = dense_layer(dec3, 500, name="output")

    with tf.name_scope("loss"):
        base_loss = tf.nn.l2_loss(X - output, name="base_loss")
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, name="reg_loss")
        total_loss = tf.add_n(base_loss, reg_loss, name="total_loss")
        # put brackets around base_loss?

    with tf.name_scope("train"):
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
        training_op = optimizer.minimize(total_loss)

    return training_op, total_loss, X, output
