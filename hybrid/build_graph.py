import tensorflow as tf
from functools import partial


# This model is untested, unused, and unloved :(
def make_small_nn(hyperparams, size_output=3, debug=False):
    with tf.variable_scope("data"):
        X = tf.compat.v1.placeholder(tf.float32, [None, hyperparams['n_inputs'] * 2], name="X")
        y = tf.compat.v1.placeholder(tf.float32, [None, size_output], name="y")
        training = tf.compat.v1.placeholder_with_default(False, shape=(), name="training")

    dense_layer = partial(tf.layers.dense, kernel_initializer=tf.contrib.layers
                          .variance_scaling_initializer(seed=hyperparams['seed']),
                          kernel_regularizer=tf.contrib.layers
                          .l2_regularizer(hyperparams['l2_scale']))
    batch_norm = partial(tf.layers.batch_normalization, training=training,
                         momentum=hyperparams['momentum'])

    size_l1 = hyperparams['size_l1']
    size_l2 = hyperparams['size_l2']
    size_lh = hyperparams['size_lh']
    size_li = hyperparams['size_li']
    # size_ouput = 3 (default)

    with tf.variable_scope("nn"):
        with tf.variable_scope("base"):
            layer1 = dense_layer(X, size_l1, name="layer1")
            layer1_activation = tf.nn.elu(batch_norm(layer1))
            layer2 = dense_layer(layer1_activation, size_l2, name="layer2")
            layer2_activation = tf.nn.elu(batch_norm(layer2))
            layerh = dense_layer(layer2_activation, size_lh, name="layerh")
            layerh_activation = tf.nn.elu(batch_norm(layerh))

        # Autoencoding branch
        with tf.variable_scope("ae"):
            layer3 = dense_layer(layerh_activation, size_l2, name="layer3")
            layer3_activation = tf.nn.elu(batch_norm(layer3))
            layer4 = dense_layer(layer3_activation, size_l1, name="layer4")
            layer4_activation = tf.nn.elu(batch_norm(layer4))
            layerout = dense_layer(layer4_activation, hyperparams['n_inputs'] * 2, name="layerout")
            layerout_activation = tf.nn.elu(batch_norm(layerout))
            ae_output = tf.identity(layerout_activation, name="output")

        # Inference branch
        with tf.variable_scope("infer"):
            layeri = dense_layer(layerh_activation, size_li, name="layeri")
            layeri_activation = tf.nn.elu(batch_norm(layeri))
            infer_output_layer = dense_layer(layeri_activation, size_output, name="output_layer")
            infer_output_layer_activation = tf.nn.elu(batch_norm(infer_output_layer))
            infer_output = tf.identity(infer_output_layer_activation, name="output")

    with tf.variable_scope("loss"):
        with tf.variable_scope("ae"):
            ae_loss_base = tf.nn.l2_loss(ae_output - X, name="loss_base")
            ae_loss_reg = tf.compat.v1.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            ae_loss_total = tf.add_n([ae_loss_base] + ae_loss_reg, name="loss_total")
        with tf.variable_scope("infer"):
            infer_loss_base = tf.nn.l2_loss(infer_output - y, name="loss_base")
            infer_loss_reg = tf.compat.v1.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            infer_loss_total = tf.add_n([infer_loss_base] + infer_loss_reg, name="loss_total")

    with tf.variable_scope("train"):
        ae_opt = tf.compat.v1.train.MomentumOptimizer(hyperparams['learning_rate'],
                                                      hyperparams['momentum'], use_nesterov=True)

        # If we want to freeze the autoencoder while training, only train on infer-scoped variables.
        # We also want to train the batch normalization terms so that the distribution of the
        #   synthetic traces can be capture in the first part of the autoencoder network.
        if hyperparams['freeze_ae']:
            infer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=".+(batch_normalization).+|.+(infer).+")
        else:
            infer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        infer_opt = tf.compat.v1.train.MomentumOptimizer(hyperparams['learning_rate'],
                                                         hyperparams['momentum'], use_nesterov=True)

        if not debug:
            ae_training_op = ae_opt.minimize(ae_loss_total)
            infer_training_op = infer_opt.minimize(infer_loss_total, infer_vars)

            return (ae_training_op, infer_training_op, X, y, training, ae_output, infer_output,
                    ae_loss_total, infer_loss_total)
        else:
            ae_grads = ae_opt.compute_gradients(ae_loss_total)
            infer_grads = infer_opt.compute_gradients(infer_loss_total, infer_vars)
            ae_training_op = ae_opt.apply_gradients(ae_grads)
            infer_training_op = infer_opt.apply_gradients(infer_grads)

            return (ae_training_op, infer_training_op, X, y, training, ae_output, infer_output,
                    ae_loss_total, infer_loss_total, ae_grads, infer_grads)


'''
def make_conv_nn(hyperparams, size_output=3, debug=False):
    with tf.name_scope("data"):
        X = tf.compat.v1.placeholder(tf.float32, [None, hyperparams['n_inputs'] * 2], name="X")
        y = tf.compat.v1.placeholder(tf.float32, [None, size_output], name="y")
        training = tf.compat.v1.placeholder_with_default(False, shape=(), name="training")

    X_reshaped = tf.reshape(X, [-1, 2, hyperparams['n_inputs'], 1])

    conv_layer = partial(tf.layers.conv2d,
                         padding='valid', activation=None,
                         kernel_initializer=tf.contrib.layers
                         .variance_scaling_initializer(seed=hyperparams['seed']),
                         kernel_regularizer=tf.contrib.layers
                         .l2_regularizer(hyperparams['scale']),
                         )

    deconv_layer = partial(tf.layers.conv2d_transpose)

    pool_layer = partial(tf.layers.max_pooling2d)

    dense_layer = partial(tf.layers.dense, kernel_initializer=tf.contrib.layers
                          .variance_scaling_initializer(seed=hyperparams['seed']),
                          kernel_regularizer=tf.contrib.layers
                          .l2_regularizer(hyperparams['scale']))
    batch_norm = partial(tf.layers.batch_normalization, training=training,
                         momentum=hyperparams['momentum'])

    min_conv_size = 468  # Determinded by the input convolutions.
    # size_ouput = 3 (default)

    with tf.name_scope("nn"):
        layer_conv0 = conv_layer(X_reshaped, name="layer_conv0", filters=32, kernel_size=(2, 5),
                                 strides=(1, 1))
        layer_conv0_activation = tf.nn.elu(batch_norm(layer_conv0))
        layer_pool0 = pool_layer(layer_conv0_activation, name="layer_poo0", pool_size=(1, 5),
                                 strides=(1, 1))

        layer_conv1 = conv_layer(layer_pool0, name="layer_conv1", filters=5, kernel_size=(1, 5),
                                 strides=(1, 1))
        layer_conv1_activation = tf.nn.elu(batch_norm(layer_conv1))
        layer_pool1 = pool_layer(layer_conv1_activation, name="layer_conv1", pool_size=(1, 5),
                                 strides=(1, 1))

        layer_conv2 = conv_layer(layer_pool1, name="layer_conv2", filters=5, kernel_size=(1, 5),
                                 strides=(1, 1))
        layer_conv2_activation = tf.nn.elu(batch_norm(layer_conv2))
        layer_pool2 = pool_layer(layer_conv2_activation, name="layer_pool2", pool_size=(1, 5),
                                 strides=(1, 1))

        layer_h0 = dense_layer(layer_pool2, name="layer_h0", hyperparams['size_lh'])
        layer_h0_activation = tf.nn.elu(batch_norm(layer_h0))

        # with tf.name_scope("deconv"):
        #     layer_deconv0 = deconv_layer(input = )

        pool_flattened = tf.reshape(WWWWWWWW, [-1, 1 * min_conv_size * 5])
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
'''
