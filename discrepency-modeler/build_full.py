import tensorflow as tf
from functools import partial
import numpy as np
from matplotlib import pyplot as plt
import sys


class Model:
    def build_data_pipeline(self, hyperparams, data_mean, data_ptp):
        with tf.variable_scope("pipeline"):
            self.data_train = tf.placeholder(tf.float32, [None, hyperparams['n_inputs'] * 2 +
                                                          hyperparams['n_flag_inputs'] +
                                                          hyperparams['n_phys_inputs'] - 2])
            self.data_test = tf.placeholder(tf.float32, [None, hyperparams['n_inputs'] * 2 +
                                                         hyperparams['n_flag_inputs'] +
                                                         hyperparams['n_phys_inputs'] - 2])

            # Keep mean and ptp in the graph so they can be accessed outside of the model.
            self.data_mean = tf.constant(data_mean, dtype=np.float32, name="data_mean")
            self.data_ptp = tf.constant(data_ptp, dtype=np.float32, name="data_ptp")
            # self.dataset = tf.data.Dataset.from_tensor_slices(self.data_input_sliced)

            self.dataset_train = tf.data.Dataset.from_tensor_slices(self.data_train)
            self.dataset_train = self.dataset_train.batch(hyperparams['batch_size'])
            self.dataset_train = self.dataset_train.repeat()
            self.dataset_train = self.dataset_train.prefetch(tf.contrib.data.AUTOTUNE)
            self.data_train_iter = self.dataset_train.make_initializable_iterator()

            self.dataset_test = tf.data.Dataset.from_tensor_slices(self.data_test)
            self.dataset_test = self.dataset_test.batch(hyperparams['batch_size'])
            self.dataset_test = self.dataset_test.repeat()
            self.dataset_test = self.dataset_test.prefetch(tf.contrib.data.AUTOTUNE)
            self.data_test_iter = self.dataset_test.make_initializable_iterator()

            # No y because this whole thing is pretty much an AE
            self.data_X_train = self.data_train_iter.get_next()
            self.data_X_test = self.data_test_iter.get_next()

    def build_CNN(self, hyperparams, X_train, X_test):
        with tf.variable_scope("data"):
            self.training = tf.compat.v1.placeholder_with_default(False, shape=(), name="training")
            self.X = tf.cond(self.training, true_fn=lambda: X_train, false_fn=lambda: X_test)
            self.X_phys = tf.identity(self.X[:, hyperparams['n_inputs'] * 2:], name="X_phys")
            self.X = tf.identity(self.X[:, 0:hyperparams['n_inputs'] * 2], name="X")
            self.X_shape = tf.shape(self.X)

            # X needs to be 4d for input into convolution layers
            self.X_reshaped = tf.reshape(self.X, [-1, 2, hyperparams['n_inputs'], 1])

        conv_layer = partial(tf.layers.conv2d, activation=None,
                             kernel_initializer=tf.contrib.layers
                             .variance_scaling_initializer(seed=hyperparams['seed']),
                             kernel_regularizer=tf.contrib.layers
                             .l2_regularizer(hyperparams['l2_CNN']),
                             )

        dense_layer = partial(tf.layers.dense, kernel_initializer=tf.contrib.layers
                              .variance_scaling_initializer(seed=hyperparams['seed']),
                              kernel_regularizer=tf.contrib.layers
                              .l2_regularizer(hyperparams['l2_CNN']))

        # pool_layer = partial(tf.layers.max_pooling2d, padding='same')
        pool_layer = partial(tf.layers.average_pooling2d, padding='same')

        batch_norm = partial(tf.layers.batch_normalization, training=self.training,
                             momentum=hyperparams['momentum'])

        # middle_size = 8
        middle_size = 4
        filters = hyperparams['filters']
        attn_filters = 8
        feat_filters = 8

        with tf.variable_scope("attn"):
            self.attn_conv0 = conv_layer(self.X_reshaped, name="attn_conv0", filters=attn_filters,
                                         kernel_size=(2, 16), strides=(2, 1), padding='same',
                                         activation=tf.nn.elu)
            self.attn_conv1 = conv_layer(batch_norm(self.attn_conv0), name="attn_conv1", filters=1,
                                         kernel_size=(1, 1), strides=(1, 1), padding='valid')
            # This soft attention mask is shape (batch_size, 1, 256, 1)
            self.attention_mask = tf.sigmoid(batch_norm(self.attn_conv1))
            self.attention_mask = tf.identity(self.attention_mask /
                                              tf.math.reduce_max(self.attention_mask, axis=2,
                                                                 keep_dims=True),
                                              name="attention_mask")

        with tf.variable_scope("feat"):
            self.feat_conv0 = conv_layer(self.X_reshaped, name="feat_conv0", filters=feat_filters,
                                         kernel_size=(2, 8), strides=(1, 1), padding='same',
                                         activation=tf.nn.elu)
            self.feat_pool0 = pool_layer(self.feat_conv0, name="feat_pool0",
                                         pool_size=(1, 8), strides=(1, 1))

            self.feat_conv1 = conv_layer(self.feat_pool0, name="feat_conv1", filters=feat_filters,
                                         kernel_size=(2, 8), strides=(1, 1), padding='same',
                                         activation=tf.nn.elu)
            self.feat_pool1 = pool_layer(self.feat_conv1, name="feat_pool1",
                                         pool_size=(1, 8), strides=(1, 1))

            # print_op = tf.print("attn_mask shape: ", self.attention_mask.shape,
            #                     "\nfeat_pool shape: ", self.feat_pool1.shape,
            #                     "\nattn_conv shape: ", self.attn_conv1.shape,
            #                     output_stream=sys.stdout)
            # with tf.control_dependencies([print_op]):
            self.attention_glimpse = self.attention_mask * self.feat_pool1

        with tf.variable_scope("nn"):
            self.layer_conv0 = conv_layer(self.attention_glimpse, name="layer_conv0",
                                          filters=filters, kernel_size=(2, 8), strides=(1, 2),
                                          padding='same', activation=tf.nn.elu)
            self.layer_pool0 = pool_layer(self.layer_conv0, name="layer_pool0",
                                          pool_size=(1, 8), strides=(1, 2))

            self.layer_conv1 = conv_layer(self.layer_pool0, name="layer_conv1",
                                          filters=filters * 2, kernel_size=(2, 8), strides=(1, 2),
                                          padding='same', activation=tf.nn.elu)
            self.layer_pool1 = pool_layer(self.layer_conv1, name="layer_pool1",
                                          pool_size=(1, 8), strides=(1, 2))

            self.layer_conv2 = conv_layer(self.layer_pool1, name="layer_conv2",
                                          filters=filters * 4, kernel_size=(2, 8), strides=(1, 2),
                                          padding='same', activation=tf.nn.elu)
            self.layer_pool2 = pool_layer(self.layer_conv2, name="layer_pool2",
                                          pool_size=(1, 8), strides=(1, 2))

            # Reshape for input into dense layers or whatever (TensorFlow needs explicit
            #   dimensions for NNs except for the batch size).
            self.conv_flattened = tf.reshape(self.layer_pool2, [-1, 2 * middle_size * filters * 4])
            # self.layer_nn1 = dense_layer(self.conv_flattened, 32)
            # self.layer_nn1_activation = tf.nn.elu(self.layer_nn1)
            # self.layer_nn2 = dense_layer(self.layer_nn1_activation, 32)
            # self.layer_nn2_activation = tf.nn.elu(self.layer_nn2)
            # self.layer_nn3 = dense_layer(self.layer_nn2_activation, 32)
            # self.layer_nn3_activation = tf.nn.elu(self.layer_nn3)
            # self.layer_nn4 = dense_layer(self.layer_nn3_activation, 32)
            # self.layer_nn4_activation = tf.nn.elu(self.layer_nn4)

            self.CNN_output = tf.identity(self.conv_flattened, name='CNN_output')

    def build_linear_translator(self, hyperparams, translator_input):
        dense_layer = partial(tf.layers.dense, kernel_initializer=tf.contrib.layers
                              .variance_scaling_initializer(seed=hyperparams['seed']),
                              kernel_regularizer=tf.contrib.layers
                              .l2_regularizer(hyperparams['l2_translator']))

        with tf.variable_scope("trans"):
            # This is the learned offset for the sweep to be applied to theory curves.
            self.layer_offset = dense_layer(translator_input, 1, name="layer_offset")
            self.layer_offset_activation = tf.identity(self.layer_offset,
                                                       name="layer_offset_activation") / 100.0
            # Divided by 100 to make it easier to learn.

            self.layer_convert = dense_layer(translator_input, hyperparams['n_phys_inputs'],
                                             name="layer_convert")
            self.layer_convert_activation = tf.identity(self.layer_convert,
                                                        name="layer_convert_activation")
            # This gets passed off to the surrogate model
            self.phys_input = self.layer_convert_activation

    def build_monoenergetic_electron_model(self, hyperparams, phys_input, vsweep, scalefactor):
        with tf.variable_scope("phys"):
            # Physical model branch
            # self.analytic_input = latent_rep

            # This is analytical simple Langmuir sweep. See generate.py for
            #   a better explanation.
            # Te is in eV in this implementation.
            S = 2e-6  # Area of the probe in m^2
            me = 9.109e-31  # Mass of electron
            e = 1.602e-19  # Elementary charge
            # Physical prefactor for the sweep equation
            physical_prefactor = e * np.sqrt(2 * e / me) * S
            # Scale the input parameters so that the network parameters are sane values. These
            #   constants were discovered empircally to enable easier training.
            # norm_Vp_factor = scalefactor[1]  # provided by surrogate model -- this is Vp
            # norm_n_p_factor = scalefactor[4]  # per m^3
            # norm_Ep_factor = scalefactor[5]  # Energy is in eV
            # norm_factors = tf.constant([norm_Vp_factor, norm_n_p_factor, norm_Ep_factor])

            mono_phys_input = tf.concat([phys_input[:, 1:2] / scalefactor[1],
                                         phys_input[:, 3:4] / scalefactor[4],
                                         phys_input[:, 4:5] / scalefactor[5]], 1)

            self.monoenergetic_input = tf.identity(mono_phys_input, name="input")
            # You need the explicit end index to preserve that dimension to enable broadcasting.
            Vp = self.monoenergetic_input[:, 0:1]
            n_p = tf.math.abs(self.monoenergetic_input[:, 1:2])
            Ep = tf.math.abs(self.monoenergetic_input[:, 2:3])
            # Lanmguir sweep calculations start here.
            Ip = n_p * tf.sqrt(Ep) * physical_prefactor

            # print_op = tf.print("Ip: ", Ip[0], "\nVp: ", Vp[0], "\nEp: ", Ep[0], "\nn_p: ", n_p[0],
                                # output_stream=sys.stdout)
            # with tf.control_dependencies([print_op]):
                # current = Ip * (1 - (Vp - vsweep) / Ep)

            current = Ip * (1 - (Vp - vsweep) / Ep)
            top_condition = tf.less(Vp, vsweep)
            # Need the _v2 to have good broadcasting support (requires TF >= 1.14).
            top_filled = tf.where_v2(top_condition, Ip, current)

            bottom_condition = tf.less_equal(vsweep, Vp - Ep)
            bottom_filled = tf.where_v2(bottom_condition, tf.constant(0.0), top_filled)

            # This is the output of the model that's given to the user.
            self.monoenergetic_output = tf.identity(bottom_filled, name="output")

    # Needed to put this in its own function because of needing to import the meta graph after
    #   building the translator. Keeps things cleaner this way, hopefully.
    def build_plasma_info(self, scalefactor):
        # Divide by some constants to get physical numbers. The analytical model has these
        #   built in so that needs to be removed if the analytical model is chosen. Only take
        #   the first three components because the 4th one is for vsweep (and it's just 1.0).
        self.plasma_info = tf.identity(self.phys_input /
                                       tf.concat([scalefactor[0:3], scalefactor[4:6]], 0),
                                       name="plasma_info")

    def build_variational_translator(self, hyperparams):
        pass

    def build_theory_processor(self, hyperparams, theory_output, stop_gradient):
        # Scale the (physical) theory output to match that of the input curves (which *was* scaled)
        scaled_theory = ((theory_output - self.data_mean[hyperparams['n_inputs']:]) /
                         self.data_ptp[hyperparams['n_inputs']:])
        # Add the learned sweep offset
        scaled_theory = scaled_theory + self.layer_offset_activation

        if stop_gradient:
            self.processed_theory = tf.stop_gradient(scaled_theory, name="processed_theory")
        else:
            self.processed_theory = tf.identity(scaled_theory, name="processed_theory")

    def build_discrepancy_model(self, hyperparams, vsweep, difference):
        dense_layer = partial(tf.layers.dense, kernel_initializer=tf.contrib.layers
                              .variance_scaling_initializer(seed=hyperparams['seed']),
                              kernel_regularizer=tf.contrib.layers
                              .l2_regularizer(hyperparams['l2_discrepancy']))

        # Start with simple dense NN for now. Should probs put batch norm in here someplace.
        with tf.variable_scope("discrepancy"):
            self.diff = tf.concat([vsweep, difference], 1)
            self.diff_layer0 = dense_layer(self.diff, hyperparams['size_diff'],
                                           name="diff_layer0")
            self.diff_layer0_activation = (self.diff_layer0)
            self.diff_layer1 = dense_layer(self.diff_layer0, hyperparams['n_output'],
                                           name="diff_layer1")
            self.diff_layer1_activation = (self.diff_layer1)
            self.discrepancy_output = tf.identity(self.diff_layer1_activation,
                                                  name="discrepancy_output")

    def build_learned_discrepancy_model(self, hyperparams, latent_rep):
        dense_layer = partial(tf.layers.dense, kernel_initializer=tf.contrib.layers
                              .variance_scaling_initializer(seed=hyperparams['seed']),
                              kernel_regularizer=tf.contrib.layers
                              .l2_regularizer(hyperparams['l2_discrepancy']))

        with tf.variable_scope("discrepancy"):
            self.diff_layer0 = dense_layer(latent_rep, hyperparams['size_diff'],
                                           name="diff_layer0")
            self.diff_layer0_activation = (self.diff_layer0)
            self.diff_layer1 = dense_layer(self.diff_layer0, hyperparams['n_output'],
                                           name="diff_layer1")
            self.diff_layer1_activation = (self.diff_layer1)
            self.discrepancy_output = tf.identity(self.diff_layer1_activation,
                                                  name="discrepancy_output")

    # Scale controls how wide the function is. The larger the scale, the larger the x-axis squish.
    def soft_sqrt(self, tensor, scale=1.0):
        absolute = scale * tf.math.abs(tensor)
        # Coefficients in front of the square root are for smoothness of the function.
        value = tf.where(absolute > 1.0,
                         x=(2.0 * tf.math.sqrt(absolute) - 1.0),
                         y=absolute)
        return value

    def sqrt(self, tensor, scale=1.0):
        absolute = scale * tf.math.abs(tensor)
        value = 2 * tf.math.sqrt(absolute + 1) - 2
        return value

    def build_loss(self, hyperparams, original, theory, discrepancy,
                   original_phys_num, scalefactor):
        with tf.variable_scope("loss"):
            loss_normalization = (hyperparams['loss_rebuilt'] + hyperparams['loss_theory'] +
                                  hyperparams['loss_discrepancy'])
            loss_scale = hyperparams['loss_scale']

            self.loss_physics = (hyperparams['loss_physics'] * 0.5 *
                                 tf.reduce_sum(tf.expand_dims(original_phys_num[:, 0], 1) *
                                               (original_phys_num[:, 1:4] *
                                                scalefactor[0:3] - self.phys_input[:, 0:3]) ** 2))

            self.loss_phys_penalty = (hyperparams['loss_phys_penalty'] *
                                      tf.reduce_sum(self.sqrt(self.phys_input, scale=loss_scale)))

            self.l1_CNN_output = (hyperparams['l1_CNN_output'] *
                                  tf.reduce_sum(tf.math.abs(self.CNN_output)))

            discrepancy = tf.constant(0.0)
            self.model_output = tf.identity(theory + discrepancy, name="model_output")
            # Penalize errors in the rebuilt trace.
            self.loss_rebuilt = (tf.reduce_sum(self.sqrt(original - self.model_output) *
                                               self.attention_mask[:, 0, :, 0],
                                               name="loss_rebuilt") *
                                 hyperparams['loss_rebuilt'] / loss_normalization)
            # Penalize errors between the theory and original trace.
            self.loss_theory = (tf.reduce_sum(self.sqrt(original - theory),
                                              name="loss_theory") *
                                hyperparams['loss_theory'] / loss_normalization)
            # Penalize the size of the discrepancy output.
            self.loss_discrepancy = 0.0
            # self.loss_discrepancy = (tf.reduce_sum(self.sqrt(discrepancy, scale=loss_scale),
            #                                        name="loss_discrepancy") *
            #                          hyperparams['loss_discrepancy'] / loss_normalization)

            # Divide model loss by batch size to keep loss consistent regardless of input size.
            self.loss_model = ((self.loss_rebuilt + self.loss_theory +
                                self.loss_discrepancy + self.loss_physics +
                                self.loss_phys_penalty + self.l1_CNN_output) /
                               tf.cast(tf.shape(self.model_output)[0], tf.float32))
            self.loss_reg = tf.compat.v1.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss_total = tf.add_n([self.loss_model] + self.loss_reg, name="loss_total")

        with tf.variable_scope("train"):
            self.opt = tf.compat.v1.train.MomentumOptimizer(hyperparams['learning_rate'],
                                                            hyperparams['momentum'],
                                                            use_nesterov=True)
            # self.opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(self.opt)
            self.grads = self.opt.compute_gradients(self.loss_total, var_list=self.vars)
            self.training_op = self.opt.apply_gradients(self.grads)

    def load_model(self, sess, model_path):
        restorer = tf.train.Saver()
        restorer.restore(sess, model_path)
        print("Model {} has been loaded.".format(model_path))

    def plot_comparison(self, sess, hyperparams, save_path, epoch):
        (model_output, theory_output, phys_numbers, data_mean, data_ptp, data_input, attn_mask
         ) = sess.run([self.model_output, self.processed_theory, self.plasma_info,
                       self.data_mean[hyperparams['n_inputs']:],
                       self.data_ptp[hyperparams['n_inputs']:],
                       self.X,
                       self.attention_mask],
                      feed_dict={self.training: False})

        batch_size = model_output.shape[0]

        data_input = data_input[:, hyperparams['n_inputs']:] * data_ptp + data_mean

        model_output = model_output * data_ptp + data_mean
        theory_output = theory_output * data_ptp + data_mean

        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8), sharex=True)
        fig.suptitle('Comparison of ')
        np.random.seed(hyperparams['seed'])
        randidx = np.random.randint(batch_size, size=(3, 4))

        # print(model_output.shape[0])

        for x, y in np.ndindex((3, 4)):
            axes[x, y].plot(data_input[randidx[x, y]], label="Data")
            axes[x, y].plot(theory_output[randidx[x, y]], label="Theory", alpha=0.8)
            # axes[x, y].plot(model_output[randidx[x, y]], label="Rebuilt", alpha=0.4)
            axes[x, y].set_title("Index {}".format(randidx[x, y]))
        axes[0, 0].legend()

        for x, y in np.ndindex((3, 4)):
            axes[x, y].text(0.05, 0.4,
                            "ne = {:3.1e} / cm$^3$ \nVp = {:.1f} V \nTe = {:.1f} eV".
                            format(phys_numbers[randidx[x, y], 0] / 1e6,
                                   phys_numbers[randidx[x, y], 1],
                                   phys_numbers[randidx[x, y], 2] / 1.602e-19) +
                            "\nnp = {:3.1e} / cm$^3$ \nEp = {:.1f} eV".
                            format(phys_numbers[randidx[x, y], 3] / 1e6,
                                   phys_numbers[randidx[x, y], 4]),
                            transform=axes[x, y].transAxes,
                            fontsize=6)
            mask_color = np.ones((attn_mask[randidx[x, y]].shape[1], 4))
            mask_color[:, 0] = 0.0
            mask_color[:, 3] = attn_mask[randidx[x, y]][0, :, 0] / 2.0
            mask_color = mask_color[np.newaxis, :, :]
            axes[x, y].imshow(mask_color, aspect='auto',
                              extent=(0.0, 256.0,
                                      axes[x, y].get_ylim()[0], axes[x, y].get_ylim()[1]))
            # print(attn_mask[randidx[x, y], 0, 127, 0])

        fig.savefig(save_path + 'full-compare-epoch-{}'.format(epoch))
        plt.close(fig)

    def __init__(self):
        pass
