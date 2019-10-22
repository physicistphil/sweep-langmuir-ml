import tensorflow as tf
from functools import partial


def make_conv_nn(hyperparams, size_output=3, debug=False):
    with tf.name_scope("data"):
        X = tf.compat.v1.placeholder(tf.float32, [None, hyperparams['n_inputs'] * 2], name="X")
        y = tf.compat.v1.placeholder(tf.float32, [None, size_output], name="y")
        training = tf.compat.v1.placeholder_with_default(False, shape=(), name="training")
        X_ptp = tf.compat.v1.placeholder(tf.float32, name="X_ptp")
        X_mean = tf.compat.v1.placeholder(tf.float32, name="X_mean")
    # X needs to be 4d for input into convolution layers
    X_reshaped = tf.reshape(X, [-1, 2, hyperparams['n_inputs'], 1])

    conv_layer = partial(tf.layers.conv2d,
                         padding='same', activation=None,
                         kernel_initializer=tf.contrib.layers
                         .variance_scaling_initializer(seed=hyperparams['seed']),
                         kernel_regularizer=tf.contrib.layers
                         .l2_regularizer(hyperparams['l2_scale']),
                         )

    upconv_layer = partial(tf.layers.conv2d_transpose,
                           padding='same', activation=None,
                           kernel_initializer=tf.contrib.layers
                           .variance_scaling_initializer(seed=hyperparams['seed']),
                           kernel_regularizer=tf.contrib.layers
                           .l2_regularizer(hyperparams['l2_scale']))

    pool_layer = partial(tf.layers.max_pooling2d, padding='same')

    dense_layer = partial(tf.layers.dense, kernel_initializer=tf.contrib.layers
                          .variance_scaling_initializer(seed=hyperparams['seed']),
                          kernel_regularizer=tf.contrib.layers
                          .l2_regularizer(hyperparams['l2_scale']))
    batch_norm = partial(tf.layers.batch_normalization, training=training,
                         momentum=hyperparams['momentum'])

    middle_size = 8
    filters = hyperparams['filters']
    # min_conv_size = 56  # Determinded by the input convolutions.
    # size_ouput = 3 (default)

    with tf.name_scope("nn"):
        with tf.variable_scope("base"):
            layer_conv0 = conv_layer(X_reshaped, name="layer_conv0", filters=filters,
                                     kernel_size=(2, 5), strides=(1, 1))
            # Just keep middle row (making the height dimension padding 'valid').
            # We need 1:2 (instaed of just 1) to preserve the dimension.
            layer_conv0_activation = tf.nn.elu(batch_norm(layer_conv0[:, :, :, :]))
            layer_pool0 = pool_layer(layer_conv0_activation, name="layer_pool0",
                                     pool_size=(1, 5), strides=(1, 2))

            layer_conv1 = conv_layer(layer_pool0, name="layer_conv1", filters=filters,
                                     kernel_size=(2, 5), strides=(1, 1))
            layer_conv1_activation = tf.nn.elu(batch_norm(layer_conv1))
            layer_pool1 = pool_layer(layer_conv1_activation, name="layer_pool1",
                                     pool_size=(1, 5), strides=(1, 2))

            layer_conv2 = conv_layer(layer_pool1, name="layer_conv2", filters=filters,
                                     kernel_size=(2, 5), strides=(1, 1))
            layer_conv2_activation = tf.nn.elu(batch_norm(layer_conv2))
            layer_pool2 = pool_layer(layer_conv2_activation, name="layer_pool2",
                                     pool_size=(1, 5), strides=(1, 2))

            layer_conv3 = conv_layer(layer_pool2, name="layer_conv3", filters=filters,
                                     kernel_size=(2, 5), strides=(1, 1))
            layer_conv3_activation = tf.nn.elu(batch_norm(layer_conv3))
            layer_pool3 = pool_layer(layer_conv3_activation, name="layer_pool3",
                                     pool_size=(1, 5), strides=(1, 2))

            layer_conv4 = conv_layer(layer_pool3, name="layer_conv4", filters=filters,
                                     kernel_size=(2, 5), strides=(1, 1))
            layer_conv4_activation = tf.nn.elu(batch_norm(layer_conv4))
            layer_pool4 = pool_layer(layer_conv4_activation, name="layer_pool4",
                                     pool_size=(1, 5), strides=(1, 2))

            layer_conv5 = conv_layer(layer_pool4, name="layer_conv5", filters=filters,
                                     kernel_size=(2, 5), strides=(1, 1))
            layer_conv5_activation = tf.nn.elu(batch_norm(layer_conv5))
            layer_pool5 = pool_layer(layer_conv5_activation, name="layer_pool5",
                                     pool_size=(1, 5), strides=(1, 2))

        # Autoencoding branch
        with tf.variable_scope("ae"):
            ae_upconv2 = upconv_layer(layer_pool5, name="layer_upconv2",
                                      kernel_size=(2, 5), filters=filters, strides=(1, 2))
            ae_upconv2_activation = tf.nn.elu(batch_norm(ae_upconv2))

            ae_upconv3 = upconv_layer(ae_upconv2_activation, name="layer_upconv3",
                                      kernel_size=(2, 5), filters=filters, strides=(1, 2))
            ae_upconv3_activation = tf.nn.elu(batch_norm(ae_upconv3))

            ae_upconv4 = upconv_layer(ae_upconv3_activation, name="layer_upconv4",
                                      kernel_size=(2, 5), filters=filters, strides=(1, 2))
            ae_upconv4_activation = tf.nn.elu(batch_norm(ae_upconv4))

            ae_upconv5 = upconv_layer(ae_upconv4_activation, name="layer_upconv5",
                                      kernel_size=(2, 5), filters=filters, strides=(1, 2),
                                      padding='same')
            ae_upconv5_activation = tf.nn.elu(batch_norm(ae_upconv5))

            ae_upconv6 = upconv_layer(ae_upconv5_activation, name="layer_upconv6",
                                      kernel_size=(2, 5), filters=filters, strides=(1, 2),
                                      padding='same')
            ae_upconv6_activation = tf.nn.elu(batch_norm(ae_upconv6))

            ae_upconv7 = upconv_layer(ae_upconv6_activation, name="layer_upconv7",
                                      kernel_size=(2, 5), filters=filters, strides=(1, 2),
                                      padding='same')
            # ae_upconv7_activation = tf.nn.elu(batch_norm(ae_upconv7))

            ae_upconv7_activation = (batch_norm(ae_upconv7))
            ae_mean = tf.reduce_mean(ae_upconv7_activation, axis=(3), keep_dims=True)

            # ae_conv_reduce = conv_layer(ae_upconv5_activation, name="ae_conv_reduce",
                                        # kernel_size=(1, 1), filters=1, strides=(1, 1))
            # ae_conv_reduce_activation = (batch_norm(ae_conv_reduce))

            # Crop the output down to match the input.
            ae_output = tf.identity(ae_mean[:, :, 7:507, :], name="output")

        # Inference branch
        norm_ne_factor = 1e17  # per m^3
        norm_Vp_factor = 10  # Vp tends to be on the order of 10
        norm_Te_factor = 1.609e-19  # J / eV
        norm_factors = [norm_ne_factor, norm_Vp_factor, norm_Te_factor]
        with tf.variable_scope("phys"):
            # Size goes: [number of examples, height * width * filters]
            phys_flattened = tf.reshape(layer_pool5, [-1, 2 * middle_size * filters])

            phys_dense0 = dense_layer(phys_flattened, hyperparams['n_phys_inputs'],
                                      name="layer_dense0")
            # Start with just linear activations
            phys_dense0_activation = (batch_norm(phys_dense0))

            phys_model_input = phys_dense0_activation * norm_factors
            # PHYISCS MODEL GOES HERE

            infer_output = tf.identity(phys_model_output, name="output")

    with tf.variable_scope("loss"):
        with tf.variable_scope("ae"):
            ae_loss_base = tf.nn.l2_loss(ae_output - X_reshaped, name="loss_base")
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