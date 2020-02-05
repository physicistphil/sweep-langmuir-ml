import tensorflow as tf
import numpy as np
from functools import partial


class Model:
    def build_data_pipeline(self, hyperparams):
        with tf.name_scope("data"):
            self.X = tf.compat.v1.placeholder(tf.float32, [None, hyperparams['n_inputs'] * 2],
                                              name="X")
            self.training = tf.compat.v1.placeholder_with_default(False, shape=(), name="training")
            self.X_ptp = tf.compat.v1.placeholder(tf.float32, name="X_ptp")
            self.X_mean = tf.compat.v1.placeholder(tf.float32, name="X_mean")

    def build_encoder(self, hyperparams, X):
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
        batch_norm = partial(tf.layers.batch_normalization, training=self.training,
                             momentum=hyperparams['momentum'])

        filters = hyperparams['filters']
        middle_size = 8
        with tf.name_scope("nn"):
            # X needs to be 4d for input into convolution layers
            self.X_reshaped = tf.reshape(X, [-1, 2, hyperparams['n_inputs'], 1])

            with tf.variable_scope("base"):
                self.layer_conv0 = conv_layer(self.X_reshaped, name="layer_conv0", filters=filters,
                                              kernel_size=(2, 5), strides=(1, 1))
                # Just keep middle row (making the height dimension padding 'valid').
                # We need 1:2 (instead of just 1) to preserve the dimension.
                self.layer_conv0_activation = tf.nn.elu(batch_norm(self.layer_conv0[:, :, :, :]))
                self.layer_pool0 = pool_layer(self.layer_conv0_activation, name="layer_pool0",
                                              pool_size=(1, 5), strides=(1, 2))

                self.layer_conv1 = conv_layer(self.layer_pool0, name="layer_conv1", filters=filters,
                                              kernel_size=(2, 5), strides=(1, 1))
                self.layer_conv1_activation = tf.nn.elu(batch_norm(self.layer_conv1))
                self.layer_pool1 = pool_layer(self.layer_conv1_activation, name="layer_pool1",
                                              pool_size=(1, 5), strides=(1, 2))

                self.layer_conv2 = conv_layer(self.layer_pool1, name="layer_conv2", filters=filters,
                                              kernel_size=(2, 5), strides=(1, 1))
                self.layer_conv2_activation = tf.nn.elu(batch_norm(self.layer_conv2))
                self.layer_pool2 = pool_layer(self.layer_conv2_activation, name="layer_pool2",
                                              pool_size=(1, 5), strides=(1, 2))

                self.layer_conv3 = conv_layer(self.layer_pool2, name="layer_conv3", filters=filters,
                                              kernel_size=(2, 5), strides=(1, 1))
                self.layer_conv3_activation = tf.nn.elu(batch_norm(self.layer_conv3))
                self.layer_pool3 = pool_layer(self.layer_conv3_activation, name="layer_pool3",
                                              pool_size=(1, 5), strides=(1, 2))

                self.layer_conv4 = conv_layer(self.layer_pool3, name="layer_conv4", filters=filters,
                                              kernel_size=(2, 5), strides=(1, 1))
                self.layer_conv4_activation = tf.nn.elu(batch_norm(self.layer_conv4))
                self.layer_pool4 = pool_layer(self.layer_conv4_activation, name="layer_pool4",
                                              pool_size=(1, 5), strides=(1, 2))

                self.layer_conv5 = conv_layer(self.layer_pool4, name="layer_conv5", filters=filters,
                                              kernel_size=(2, 5), strides=(1, 1))
                self.layer_conv5_activation = tf.nn.elu(batch_norm(self.layer_conv5))
                self.layer_pool5 = pool_layer(self.layer_conv5_activation, name="layer_pool5",
                                              pool_size=(1, 5), strides=(1, 2))
                # This first portion is a NN layer to translate the latent space representation to
                #   a physical one.
                # Size goes: [number of examples, height * width * filters]
                self.phys_flattened = tf.reshape(self.layer_pool5, [-1, 2 * middle_size * filters])
                self.phys_dense0 = dense_layer(self.phys_flattened, hyperparams['n_phys_inputs'],
                                               name="layer_dense0")
                # I've found that elu works best.
                self.phys_dense0_activation = tf.nn.elu(self.phys_dense0)
                # Now pass off to analytical model

    # Analytic input is size [batch_size, n_phys_inputs] with values of order 1.
    # vsweep unscaled is the original values of the vsweep.
    def build_analytical_model(self, hyperparams, analytic_input, vsweep_unscaled):
        with tf.variable_scope("phys"):
            # Physical model branch
            # Constrain to guarantee positive numbers (or else NaNs appear from sqrt).
            epsilon = 1e-12
            self.analytic_input = analytic_input + epsilon

            # This is analytical simple Langmuir sweep. See generate.py for
            #   a better explanation.
            # Te is in eV in this implementation.
            S = 2e-6  # Area of the probe in m^2
            me = 9.109e-31  # Mass of electron
            e = 1.602e-19  # Elementary charge
            # Physical prefactor for the sweep equation
            physical_prefactor = (e ** (3 / 2)) / np.sqrt(2 * np.pi * me)
            # Scale the input parameters so that the network parameters are sane values. These
            #   constants were discovered empircally to enable easier training.
            norm_ne_factor = 1e17  # per m^3
            norm_Vp_factor = 20  # Vp tends to be on the order of 10--this should be fine
            norm_Te_factor = 5  # Temperature is in eV
            norm_factors = [norm_ne_factor, norm_Vp_factor, norm_Te_factor]
            self.phys_input = tf.identity(self.analytic_input * norm_factors, name="input")
            # You need the explicit end index to preserve that dimension to enable broadcasting.
            ne = self.phys_input[:, 0:1]
            Vp = self.phys_input[:, 1:2]
            Te = self.phys_input[:, 2:3]
            # Lanmguir sweep calculations start here.
            I_esat = S * ne * physical_prefactor
            # Need the X_ptp and X_mean to scale the vsweep to original values.
            # vsweep = (X[:, 0:hyperparams['n_inputs']] * X_ptp[0:hyperparams['n_inputs']] +
            # X_mean[0:hyperparams['n_inputs']])
            vsweep = vsweep_unscaled
            # My god do exponents screw up gradient descent.
            current = (I_esat * tf.sqrt(Te) * tf.exp(-(Vp - vsweep) / Te))
            esat_condition = tf.less(Vp, vsweep)
            # Need the _v2 to have good broadcasting support (requires TF >= 1.14).
            esat_filled = tf.where_v2(esat_condition, I_esat * tf.sqrt(Te), current)

            # This is the output of the model that's given to the user.
            self.phys_output = tf.identity(esat_filled, name="output")

    def build_loss(self, hyperparams, phys_output, current, current_mean, current_ptp):
        with tf.variable_scope("loss"):
            # Output to optimize on. Scale to match input.
            self.phys_output_scaled = ((phys_output - current_mean) / current_ptp)
            self.phys_loss_base = tf.nn.l2_loss(self.phys_output_scaled - current,
                                                name="loss_base")
            self.phys_loss_reg = tf.compat.v1.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.phys_loss_total = tf.add_n([self.phys_loss_base] + self.phys_loss_reg,
                                            name="loss_total")

        with tf.variable_scope("train"):
            self.phys_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.phys_opt = tf.compat.v1.train.MomentumOptimizer(hyperparams['learning_rate'],
                                                                 hyperparams['momentum'],
                                                                 use_nesterov=True)

            self.phys_grads, self.pvars = zip(*self.phys_opt.compute_gradients(self.phys_loss_total,
                                                                               self.phys_vars))
            # Gradient clipping seems to be an absolute necessity for training with an exp function.
            self.phys_grads, _ = tf.clip_by_global_norm(self.phys_grads, 1.0)

            self.phys_training_op = self.phys_opt.apply_gradients(zip(self.phys_grads, self.pvars))

    def __init__(self):
        pass
