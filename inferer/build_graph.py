import tensorflow as tf
from functools import partial


def make_small_nn(hyperparams, size_output=3, debug=False):
    with tf.name_scope("data"):
        X = tf.compat.v1.placeholder(tf.float32, [None, hyperparams['n_inputs'] * 2], name="X")
        y = tf.compat.v1.placeholder(tf.float32, [None, size_output], name="y")
        training = tf.compat.v1.placeholder_with_default(False, shape=(), name="training")

    dense_layer = partial(tf.layers.dense, kernel_initializer=tf.contrib.layers
                          .variance_scaling_initializer(seed=hyperparams['seed']),
                          kernel_regularizer=tf.contrib.layers
                          .l2_regularizer(hyperparams['scale']))
    batch_norm = partial(tf.layers.batch_normalization, training=training,
                         momentum=hyperparams['momentum'])

    size_l1 = hyperparams['size_l1']
    size_l2 = hyperparams['size_l2']
    # size_ouput = 3 (default)

    with tf.name_scope("nn"):
        layer1 = dense_layer(X, size_l1, name="layer1")
        layer1_activation = tf.nn.elu(batch_norm(layer1))
        layer2 = dense_layer(layer1_activation, size_l2, name="layer2")
        layer2_activation = tf.nn.elu(batch_norm(layer2))
        output_layer = dense_layer(layer2_activation, size_output, name="output_layer")
        output_layer_activation = tf.nn.elu(batch_norm(output_layer))
        output = tf.identity(output_layer_activation, name="output")

    with tf.name_scope("loss"):
        loss_base = tf.nn.l2_loss(output - y, name="loss_base")
        loss_reg = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        loss_total = tf.add_n([loss_base] + loss_reg, name="loss_total")

    with tf.name_scope("train"):
        optimizer = tf.compat.v1.train.MomentumOptimizer(hyperparams['learning_rate'],
                                                         hyperparams['momentum'], use_nesterov=True)

        if not debug:
            training_op = optimizer.minimize(loss_total)
            return training_op, X, y, training, output, loss_total
        else:
            grads = optimizer.compute_gradients(loss_total)
            training_op = optimizer.apply_gradients(grads)
            return training_op, X, y, training, output, loss_total, grads


def make_conv_nn(hyperparams, size_output=3, debug=False):
    with tf.name_scope("data"):
        X = tf.compat.v1.placeholder(tf.float32, [None, hyperparams['n_inputs'] * 2], name="X")
        y = tf.compat.v1.placeholder(tf.float32, [None, size_output], name="y")
        training = tf.compat.v1.placeholder_with_default(False, shape=(), name="training")

    X_reshaped = tf.reshape(X, [-1, 2, hyperparams['n_inputs'], 1])

    # conv1 = tf.layers.conv2d(inputs=X_reshaped)

    conv_layer = partial(tf.layers.conv2d,
                         padding='valid', activation=None,
                         kernel_initializer=tf.contrib.layers
                         .variance_scaling_initializer(seed=hyperparams['seed']),
                         kernel_regularizer=tf.contrib.layers
                         .l2_regularizer(hyperparams['scale']),
                         )

    # pool_layer = partial(tf.layers.average_pooling2d)
    pool_layer = partial(tf.layers.max_pooling2d)

    dense_layer = partial(tf.layers.dense, kernel_initializer=tf.contrib.layers
                          .variance_scaling_initializer(seed=hyperparams['seed']),
                          kernel_regularizer=tf.contrib.layers
                          .l2_regularizer(hyperparams['scale']))
    batch_norm = partial(tf.layers.batch_normalization, training=training,
                         momentum=hyperparams['momentum'])

    # size_l1 = hyperparams['size_l1']
    # size_l2 = hyperparams['size_l2']
    # size_ouput = 3 (default)

    with tf.name_scope("nn"):
        layer_conv0 = conv_layer(X_reshaped, name="layer_conv0", filters=64, kernel_size=(2, 5),
                                 strides=(1, 1))
        layer_conv0_activation = tf.nn.elu(batch_norm(layer_conv0), name="layer_con0_activation")
        layer_pool0 = pool_layer(layer_conv0_activation, name="layer_poo0", pool_size=(1, 5),
                                 strides=(1, 1))

        layer_conv1 = conv_layer(layer_pool0, name="layer_conv1", filters=5, kernel_size=(1, 10),
                                 strides=(1, 5))
        layer_conv1_activation = tf.nn.elu(batch_norm(layer_conv1))
        layer_pool1 = pool_layer(layer_conv1_activation, name="layer_conv1", pool_size=(1, 10),
                                 strides=(1, 5))

        # layer1 = dense_layer(layer_conv, size_l1, name="layer1")
        # layer1_activation = tf.nn.elu(batch_norm(layer1))

        # layer2 = dense_layer(layer1_activation, size_l2, name="layer2")
        # layer2_activation = tf.nn.elu(batch_norm(layer2))
        pool_flattened = tf.reshape(layer_pool1, [-1, 1 * 18 * 5])

        output_layer = dense_layer(pool_flattened, size_output, name="output_layer")
        output_layer_activation = tf.nn.elu(batch_norm(output_layer))
        output = tf.identity(output_layer_activation, name="output")

    with tf.name_scope("loss"):
        loss_base = tf.nn.l2_loss(output - y, name="loss_base")
        loss_reg = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        loss_total = tf.add_n([loss_base] + loss_reg, name="loss_total")

    with tf.name_scope("train"):
        optimizer = tf.compat.v1.train.MomentumOptimizer(hyperparams['learning_rate'],
                                                         hyperparams['momentum'], use_nesterov=True)

        if not debug:
            training_op = optimizer.minimize(loss_total)
            return training_op, X, y, training, output, loss_total
        else:
            grads = optimizer.compute_gradients(loss_total)
            training_op = optimizer.apply_gradients(grads)
            return training_op, X, y, training, output, loss_total, grads
