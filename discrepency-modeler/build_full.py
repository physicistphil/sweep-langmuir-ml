import tensorflow as tf
from functools import partial
import numpy as np
from matplotlib import pyplot as plt


class Model:
    def build_data_pipeline(self, hyperparams, data_mean, data_ptp):
        with tf.variable_scope("pipeline"):
            self.data_train = tf.placeholder(tf.float32, [None, hyperparams['n_inputs'] * 2 +
                                                          hyperparams['n_flag_inputs'] +
                                                          hyperparams['n_phys_inputs']])
            self.data_test = tf.placeholder(tf.float32, [None, hyperparams['n_inputs'] * 2 +
                                                         hyperparams['n_flag_inputs'] +
                                                         hyperparams['n_phys_inputs']])

            # Keep mean and ptp in the graph so they can be accessed outside of the model.
            self.data_mean = tf.constant(data_mean, dtype=np.float32, name="data_mean")
            self.data_ptp = tf.constant(data_ptp, dtype=np.float32, name="data_ptp")
            # self.dataset = tf.data.Dataset.from_tensor_slices(self.data_input_sliced)

            self.dataset_train = tf.data.Dataset.from_tensor_slices(self.data_train)
            self.dataset_train = self.dataset_train.batch(hyperparams['batch_size'])
            self.dataset_train = self.dataset_train.repeat()
            self.dataset_train = self.dataset_train.apply(tf.data.experimental.
                                                          copy_to_device("/gpu:0"))
            self.dataset_train = self.dataset_train.prefetch(tf.contrib.data.AUTOTUNE)
            self.data_train_iter = self.dataset_train.make_initializable_iterator()

            self.dataset_test = tf.data.Dataset.from_tensor_slices(self.data_test)
            self.dataset_test = self.dataset_test.batch(hyperparams['batch_size'])
            self.dataset_test = self.dataset_test.repeat()
            self.dataset_test = self.dataset_test.apply(tf.data.experimental.
                                                        copy_to_device("/gpu:0"))
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

        conv_layer = partial(tf.layers.conv2d,
                             padding='same', activation=None,
                             kernel_initializer=tf.contrib.layers
                             .variance_scaling_initializer(seed=hyperparams['seed']),
                             kernel_regularizer=tf.contrib.layers
                             .l2_regularizer(hyperparams['l2_CNN']),
                             )

        pool_layer = partial(tf.layers.max_pooling2d, padding='same')

        batch_norm = partial(tf.layers.batch_normalization, training=self.training,
                             momentum=hyperparams['momentum'])

        middle_size = 8
        filters = hyperparams['filters']

        with tf.variable_scope("nn"):
            self.layer_conv0 = conv_layer(self.X_reshaped, name="layer_conv0", filters=filters,
                                          kernel_size=(2, 5), strides=(1, 1))
            # Just keep middle row (making the height dimension padding 'valid').
            # We need 1:2 (instead of just 1) to preserve the dimension.
            self.layer_conv0_activation = tf.nn.elu((self.layer_conv0[:, :, :, :]))
            self.layer_pool0 = pool_layer(self.layer_conv0_activation, name="layer_pool0",
                                          pool_size=(1, 5), strides=(1, 2))

            self.layer_conv1 = conv_layer(self.layer_pool0, name="layer_conv1", filters=filters,
                                          kernel_size=(2, 5), strides=(1, 1))
            self.layer_conv1_activation = tf.nn.elu((self.layer_conv1))
            self.layer_pool1 = pool_layer(self.layer_conv1_activation, name="layer_pool1",
                                          pool_size=(1, 5), strides=(1, 2))

            self.layer_conv2 = conv_layer(self.layer_pool1, name="layer_conv2", filters=filters,
                                          kernel_size=(2, 5), strides=(1, 1))
            self.layer_conv2_activation = tf.nn.elu((self.layer_conv2))
            self.layer_pool2 = pool_layer(self.layer_conv2_activation, name="layer_pool2",
                                          pool_size=(1, 5), strides=(1, 2))

            self.layer_conv3 = conv_layer(self.layer_pool2, name="layer_conv3", filters=filters,
                                          kernel_size=(2, 5), strides=(1, 1))
            self.layer_conv3_activation = tf.nn.elu((self.layer_conv3))
            self.layer_pool3 = pool_layer(self.layer_conv3_activation, name="layer_pool3",
                                          pool_size=(1, 5), strides=(1, 2))

            self.layer_conv4 = conv_layer(self.layer_pool3, name="layer_conv4", filters=filters,
                                          kernel_size=(2, 5), strides=(1, 1))
            self.layer_conv4_activation = tf.nn.elu((self.layer_conv4))
            self.layer_pool4 = pool_layer(self.layer_conv4_activation, name="layer_pool4",
                                          pool_size=(1, 5), strides=(1, 2))

            # Reshape for input into dense layers or whatever (TensorFlow needs explicit
            #   dimensions for NNs except for the batch size).
            self.conv_flattened = tf.reshape(self.layer_pool4, [-1, 2 * middle_size * filters])
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

    # Needed to put this in its own function because of needing to import the meta graph after
    #   building the translator. Keeps things cleaner this way, hopefully.
    def build_plasma_info(self, scalefactor):
        # Divide by some constants to get physical numbers. The analytical model has these
        #   built in so that needs to be removed if the analytical model is chosen. Only take
        #   the first three components because the last one is for vsweep (and it's just 1.0).
        self.plasma_info = tf.identity(self.phys_input / scalefactor[0:3], name="plasma_info")

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

    def build_loss(self, hyperparams, original, theory, discrepancy,
                   original_phys_num, scalefactor):
        with tf.variable_scope("loss"):
            loss_normalization = (hyperparams['loss_rebuilt'] + hyperparams['loss_theory'] +
                                  hyperparams['loss_discrepancy'])

            self.loss_physics = (hyperparams['loss_physics'] * 0.5 *
                                 tf.reduce_sum(tf.expand_dims(original_phys_num[:, 0], 1) *
                                               (original_phys_num[:, 1:4] *
                                                scalefactor[0:3] - self.phys_input) ** 2))

            self.loss_phys_penalty = (hyperparams['loss_phys_penalty'] *
                                      tf.reduce_sum(self.soft_sqrt(self.phys_input, scale=10.0)))

            self.l1_CNN_output = (hyperparams['l1_CNN_output'] *
                                  tf.reduce_sum(tf.math.abs(self.CNN_output)))

            self.model_output = tf.identity(theory + discrepancy, name="model_output")
            # Penalize errors in the rebuilt trace.
            self.loss_rebuilt = (tf.reduce_sum(self.soft_sqrt(original - self.model_output),
                                               name="loss_rebuilt") *
                                 hyperparams['loss_rebuilt'] / loss_normalization)
            # Penalize errors between the theory and original trace.
            self.loss_theory = (tf.reduce_sum(self.soft_sqrt(original - theory),
                                              name="loss_theory") *
                                hyperparams['loss_theory'] / loss_normalization)
            # Penalize the size of the discrepancy output.
            self.loss_discrepancy = (tf.reduce_sum(self.soft_sqrt(discrepancy, scale=5.0),
                                                   name="loss_discrepancy") *
                                     hyperparams['loss_discrepancy'] / loss_normalization)

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
        (model_output, theory_output, phys_numbers, data_mean, data_ptp, data_input
         ) = sess.run([self.model_output, self.processed_theory, self.plasma_info,
                       self.data_mean[hyperparams['n_inputs']:],
                       self.data_ptp[hyperparams['n_inputs']:],
                       self.X],
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
            axes[x, y].plot(model_output[randidx[x, y]], label="Rebuilt", alpha=0.4)
            axes[x, y].set_title("Index {}".format(randidx[x, y]))
        axes[0, 0].legend()

        for x, y in np.ndindex((3, 4)):
            axes[x, y].text(0.05, 0.7,
                            "ne = {:3.1e} / cm$^3$ \nVp = {:.1f} V \nTe = {:.1f} eV".
                            format(phys_numbers[randidx[x, y], 0] / 1e6,
                                   phys_numbers[randidx[x, y], 1],
                                   phys_numbers[randidx[x, y], 2] / 1.602e-19),
                            transform=axes[x, y].transAxes)

        fig.savefig(save_path + 'full-compare-epoch-{}'.format(epoch))
        plt.close(fig)

    def __init__(self):
        pass
