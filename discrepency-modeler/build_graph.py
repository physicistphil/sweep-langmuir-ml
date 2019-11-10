import tensorflow as tf
import numpy as np
from functools import partial


def make_phys_nn(hyperparams):
    with tf.name_scope("data"):
        X = tf.compat.v1.placeholder(tf.float32, [None, hyperparams['n_inputs'] * 2], name="X")
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
    # min_conv_size = 56  # Determined by the input convolutions.
    # size_ouput = 3 (default)

    with tf.name_scope("nn"):
        with tf.variable_scope("base"):
            layer_conv0 = conv_layer(X_reshaped, name="layer_conv0", filters=filters,
                                     kernel_size=(2, 5), strides=(1, 1))
            # Just keep middle row (making the height dimension padding 'valid').
            # We need 1:2 (instead of just 1) to preserve the dimension.
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

        # Physical model branch
        S = 2e-6  # Area of the probe
        me = 9.109e-31  # Mass of electron
        e = 1.602e-19  # Elementary charge
        norm_ne_factor = 1e17  # per m^3
        norm_Vp_factor = 10  # Vp tends to be on the order of 10
        norm_Te_factor = 1.609e-19  # J / eV
        norm_factors = [norm_ne_factor, norm_Vp_factor, norm_Te_factor]
        with tf.variable_scope("phys"):
            # Size goes: [number of examples, height * width * filters]
            phys_flattened = tf.reshape(layer_pool5, [-1, 2 * middle_size * filters])
            phys_dense0 = dense_layer(phys_flattened, hyperparams['n_phys_inputs'],
                                      name="layer_dense0")
            # Add the 1 to the ELU to guarantee positive numbers (or else NaNs appear).
            phys_dense0_activation = tf.nn.elu((phys_dense0)) + 1

            # This is analytical simple Langmuir sweep. See generate.py for a better explanation.
            # Scale the input parameters so that the network parameters are sane values,
            #   but use the computed values so that it cannot hallucinate a different trace (maybe?)
            phys_input = tf.identity(phys_dense0_activation * 1, name="input")
            # Lanmguir sweep calculations.
            # You need the explicit end index to preserve that dimension to enable broadcasting.
            I_esat = S * phys_input[:, 0:1] * e / np.sqrt(2 * np.pi * me)
            current = (I_esat * tf.sqrt(phys_input[:, 2:3]) *
                       tf.exp(-e * (phys_input[:, 1:2] - X[:, 0:hyperparams['n_inputs']]) /
                              phys_input[:, 2:3]))
            esat_condition = tf.less(phys_input[:, 1:2], X[:, 0:hyperparams['n_inputs']])
            # Need the _v2 to have good broadcasting support (requires TF > 1.14).
            esat_filled = tf.where_v2(esat_condition, current, I_esat)

            # phys_output = tf.Variable(name="output")
            # phys_output.assign(esat_location, )
            # current.assign(esat_location)

            # Output to optimize on.
            phys_output = tf.identity(esat_filled, name="output")

    with tf.variable_scope("loss"):
        with tf.variable_scope("ae"):
            ae_loss_base = tf.nn.l2_loss(ae_output - X_reshaped, name="loss_base")
            ae_loss_reg = tf.compat.v1.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            ae_loss_total = tf.add_n([ae_loss_base] + ae_loss_reg, name="loss_total")
        with tf.variable_scope("phys"):
            phys_loss_base = tf.nn.l2_loss(phys_output - X[:, hyperparams['n_inputs']:] *
                                           X_ptp[hyperparams['n_inputs']:] +
                                           X_mean[hyperparams['n_inputs']:], name="loss_base")
            phys_loss_reg = tf.compat.v1.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            phys_loss_total = tf.add_n([phys_loss_base] + phys_loss_reg, name="loss_total")

    with tf.variable_scope("train"):
        ae_opt = tf.compat.v1.train.MomentumOptimizer(hyperparams['learning_rate'],
                                                      hyperparams['momentum'], use_nesterov=True)

        # If we want to freeze the autoencoder while training, only train on infer-scoped variables.
        # We also want to train the batch normalization terms so that the distribution of the
        #   synthetic traces can be capture in the first part of the autoencoder network.
        if hyperparams['freeze_ae']:
            phys_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope=".+(batch_normalization).+|.+(phys).+")
        else:
            phys_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        phys_opt = tf.compat.v1.train.MomentumOptimizer(hyperparams['learning_rate'],
                                                        hyperparams['momentum'], use_nesterov=True)

        ae_grads = ae_opt.compute_gradients(ae_loss_total)

        phys_grads, pvars = zip(*phys_opt.compute_gradients(phys_loss_total, phys_vars))
        phys_grads, _ = tf.clip_by_global_norm(phys_grads, 1.0)

        ae_training_op = ae_opt.apply_gradients(ae_grads)
        phys_training_op = phys_opt.apply_gradients(zip(phys_grads, pvars))

        return (ae_training_op, phys_training_op, X, X_mean, X_ptp, training, ae_output,
                phys_output, ae_loss_total, phys_loss_total, ae_grads, zip(phys_grads, pvars))
