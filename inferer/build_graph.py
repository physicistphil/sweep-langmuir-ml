import tensorflow as tf
from functools import partial


def make_small_nn(hyperparams, size_output=3, debug=False):
    with tf.name_scope("data"):
        X = tf.placeholder(tf.float32, [None, hyperparams['n_inputs']], name="X")
        y = tf.placeholder(tf.float32, [None, size_output], name="y")
        training = tf.placeholder_with_default(False, shape=(), name="training")
        # normalize desired outputs
        mean = tf.reduce_mean(y, axis=0)
        diff = tf.reduce_max(y, axis=0) - tf.reduce_min(y, axis=0)
        y_normalized = (y - mean) / diff

    dense_layer = partial(tf.layers.dense, kernel_initializer=tf.contrib.layers
                          .variance_scaling_initializer(seed=hyperparams['seed']),
                          kernel_regularizer=tf.contrib.layers
                          .l2_regularizer(hyperparams['scale']))
    batch_norm = partial(tf.layers.batch_normalization, training=training,
                         momentum=hyperparams['momentum'])

    size_l1 = 100
    size_l2 = 20

    with tf.name_scope("nn"):
        layer1 = dense_layer(X, size_l1, name="layer1")
        layer1_activation = tf.nn.elu(batch_norm(layer1))
        layer2 = dense_layer(layer1_activation, size_l2, name="layer2")
        layer2_activation = tf.nn.elu(batch_norm(layer2))
        output_layer = dense_layer(layer2_activation, size_output, name="output_layer")
        output_layer_activation = tf.nn.elu(batch_norm(output_layer))
        output = tf.identity(output_layer_activation, name="output")
        output_scaled = (output * diff) + mean

    with tf.name_scope("loss"):
        loss_base = tf.nn.l2_loss(output - y_normalized, name="loss_base")
        loss_reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_total = tf.add_n([loss_base] + loss_reg, name="loss_total")

    with tf.name_scope("train"):
        optimizer = tf.train.MomentumOptimizer(hyperparams['learning_rate'],
                                               hyperparams['momentum'], use_nesterov=True)

        if not debug:
            training_op = optimizer.minimize(loss_total)
            return training_op, X, y, training, output_scaled, loss_total
        else:
            grads = optimizer.compute_gradients(loss_total)
            training_op = optimizer.apply_gradients(grads)
            return training_op, X, y, training, output_scaled, loss_total, grads
