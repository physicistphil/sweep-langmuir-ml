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

        with tf.variable_scope("phys"):
            # Physical model branch
            # This first portion is a NN layer to translate the latent space representation to a
            #   physical one.
            # Size goes: [number of examples, height * width * filters]
            phys_flattened = tf.reshape(layer_pool5, [-1, 2 * middle_size * filters])
            phys_dense0 = dense_layer(phys_flattened, hyperparams['n_phys_inputs'],
                                      name="layer_dense0")
            # Constrain to guarantee positive numbers (or else NaNs appear from sqrt).
            epsilon = 1e-12
            # I've found that elu works best.
            phys_dense0_activation = tf.nn.elu(phys_dense0) + epsilon

            # This is analytical simple Langmuir sweep. See generate.py for a better explanation.
            # Te is in eV in this implementation.
            S = 2e-6  # Area of the probe in m^2
            me = 9.109e-31  # Mass of electron
            e = 1.602e-19  # Elementary charge
            # Physical prefactor for the sweep equation
            physical_prefactor = (e ** (3 / 2)) / np.sqrt(2 * np.pi * me)
            # Scale the input parameters so that the network parameters are sane values.
            norm_ne_factor = 1e17  # per m^3
            norm_Vp_factor = 20  # Vp tends to be on the order of 10--this should be fine
            norm_Te_factor = 5  # Temperature is in eV
            norm_factors = [norm_ne_factor, norm_Vp_factor, norm_Te_factor]
            phys_input = tf.identity(phys_dense0_activation * norm_factors, name="input")
            # You need the explicit end index to preserve that dimension to enable broadcasting.
            ne = phys_input[:, 0:1]
            Vp = phys_input[:, 1:2]
            Te = phys_input[:, 2:3]
            # Lanmguir sweep calculations start here.
            I_esat = S * ne * physical_prefactor
            # Need the X_ptp and X_mean to scale the vsweep to original values.
            vsweep = (X[:, 0:hyperparams['n_inputs']] * X_ptp[0:hyperparams['n_inputs']] +
                      X_mean[0:hyperparams['n_inputs']])
            # My god do exponents screw up gradient descent.
            current = (I_esat * tf.sqrt(Te) * tf.exp(-(Vp - vsweep) / Te))
            esat_condition = tf.less(Vp, vsweep)
            # Need the _v2 to have good broadcasting support (requires TF >= 1.14).
            esat_filled = tf.where_v2(esat_condition, I_esat * tf.sqrt(Te), current)

            # This is the output given to the user.
            phys_output = tf.identity(esat_filled, name="output")
            # Output to optimize on. Scale to match input.
            phys_output_scaled = ((phys_output - X_mean[hyperparams['n_inputs']:]) /
                                  X_ptp[hyperparams['n_inputs']:])

        with tf.variable_scope("phys"):
            phys_loss_base = tf.nn.l2_loss(phys_output_scaled - X[:, hyperparams['n_inputs']:],
                                           name="loss_base")
            phys_loss_reg = tf.compat.v1.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            phys_loss_total = tf.add_n([phys_loss_base] + phys_loss_reg, name="loss_total")

    with tf.variable_scope("train"):
        phys_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        phys_opt = tf.compat.v1.train.MomentumOptimizer(hyperparams['learning_rate'],
                                                        hyperparams['momentum'], use_nesterov=True)

        phys_grads, pvars = zip(*phys_opt.compute_gradients(phys_loss_total, phys_vars))
        # Gradient clipping seems to be an absolute necessity for training with an exp function.
        phys_grads, _ = tf.clip_by_global_norm(phys_grads, 1.0)

        phys_training_op = phys_opt.apply_gradients(zip(phys_grads, pvars))

        return (phys_training_op, X, X_mean, X_ptp, training, phys_output, phys_loss_total,
                zip(phys_grads, pvars), phys_input)
