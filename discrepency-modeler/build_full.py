import tensorflow as tf
from functools import partial
import numpy as np
from matplotlib import pyplot as plt


class Model:
    def build_data_pipeline(self, hyperparams):
        with tf.name_scope("pipeline"):
            self.data_input = tf.placeholder(tf.float32, [None, hyperparams['n_inputs'] * 2])
            # The data has 2 examples at the end that are actually the mean and ptp. We need to
            #   trim them off:
            self.data_mean = self.data_input[-2, :]
            self.data_ptp = self.data_input[-1, :]
            self.data_input_sliced = self.data_input[:-2, :]

            self.dataset = tf.data.Dataset.from_tensor_slices(self.data_input_sliced)
            self.dataset = self.dataset.repeat()
            self.dataset = self.dataset.batch(hyperparams['batch_size'])
            self.dataset = self.dataset.prefetch(4)

            self.data_iter = self.dataset.make_initializable_iterator()
            # No y because this whole thing is pretty much an AE
            self.data_X = self.data_iter.get_next()

    def build_CNN(self, hyperparams, X):
        with tf.name_scope("data"):
            self.training = tf.compat.v1.placeholder_with_default(False, shape=(), name="training")

            # X needs to be 4d for input into convolution layers
            self.X_reshaped = tf.reshape(X, [-1, 2, hyperparams['n_inputs'], 1])

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

        with tf.name_scope("nn"):
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
            self.layer_convert = dense_layer(translator_input, hyperparams['n_phys_inputs'],
                                             name="layer_convert")
            self.layer_convert_activation = tf.nn.elu(self.layer_convert,
                                                      name="layer_convert_activation")
            # This gets passed off to the surrogate model.

    def build_variational_translator(self, hyperparams):
        pass

    def build_theory_processor(self, hyperparams, theory_output, stop_gradient=True):
        if stop_gradient:
            self.processed_theory = tf.stop_gradient(theory_output)
        else:
            self.processed_theory = tf.identity(theory_output)

    # self.diff = X[:, hyperparams['n_inputs']:] - theory_output
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
            self.diff_layer0_activation = tf.nn.elu(self.diff_layer0)
            self.diff_layer1 = dense_layer(self.diff_layer0, hyperparams['n_output'],
                                           name="diff_layer1")
            self.diff_layer1_activation = tf.nn.elu(self.diff_layer1)
            self.discrepancy_output = tf.identity(self.diff_layer1_activation,
                                                  name="discrepancy_output")

    # discrepancy
    def build_loss(self, hyperparams, original, theory, discrepancy):
        with tf.variable_scope("loss"):
            loss_normalization = (hyperparams['loss_rebuilt'] * hyperparams['loss_theory'] *
                                  hyperparams['loss_discrepancy'])

            self.model_output = theory + discrepancy
            # Penalize errors in the rebuilt trace.
            self.loss_rebuilt = (tf.nn.l2_loss(original - self.model_output, name="loss_rebuilt") *
                                 hyperparams['loss_rebuilt'] / loss_normalization)
            # Penalize errors between the theory and original trace.
            self.loss_theory = (tf.nn.l2_loss(original - theory, name="loss_theory") *
                                hyperparams['loss_theory'] / loss_normalization)
            # Penalize the size of the discrepancy output.
            self.loss_discrepancy = (tf.nn.l2_loss(discrepancy, name="loss_discrepancy") *
                                     hyperparams['loss_discrepancy'] / loss_normalization)

            self.loss_reg = tf.compat.v1.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss_total = tf.add_n([self.loss_rebuilt + self.loss_theory +
                                        self.loss_discrepancy] + self.loss_reg,
                                       name="loss_total")

        with tf.variable_scope("train"):
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.opt = tf.compat.v1.train.MomentumOptimizer(hyperparams['learning_rate'],
                                                            hyperparams['momentum'],
                                                            use_nesterov=True)

            self.grads = self.opt.compute_gradients(self.loss_total)
            self.training_op = self.opt.apply_gradients(self.grads)

    def load_model(self, sess, model_path):
        restorer = tf.train.Saver()
        restorer.restore(sess, model_path)
        print("Model {} has been loaded.".format(model_path))

    def plot_comparison(self, sess, phys_model, data_input, hyperparams, save_path, epoch):
        # print("Plotting comparison")

        # Shape of nn_output is [?, ?]
        (model_output, theory_output,
         phys_numbers) = sess.run([self.model_output, self.processed_theory,
                                   phys_model.phys_input],
                                  feed_dict={self.training: False,
                                             self.data_input: data_input})
        # Last two "examples" are mean and ptp. Take last half of sweep for just the current.
        data_mean = data_input[-2, hyperparams['n_inputs']:]
        data_ptp = data_input[-1, hyperparams['n_inputs']:]
        data_input = data_input[:-2, hyperparams['n_inputs']:] * data_ptp + data_mean

        model_output = model_output * data_ptp + data_mean

        # generated_trace = output_test
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8), sharex=True)
        fig.suptitle('Comparison of ')
        np.random.seed(hyperparams['seed'])
        randidx = np.random.randint(model_output.shape[0], size=(3, 4))

        for x, y in np.ndindex((3, 4)):
            axes[x, y].plot(data_input[randidx[x, y]], label="Data")
            axes[x, y].plot(theory_output[randidx[x, y]], label="Theory")
            axes[x, y].plot(model_output[randidx[x, y]], label="Rebuilt")
            axes[x, y].set_title("Index {}".format(randidx[x, y]))
        axes[0, 0].legend()

        for x, y in np.ndindex((3, 4)):
            axes[x, y].text(0.05, 0.7,
                            "ne = {:3.1e} / cm$^3$ \nVp = {:.1f} V \nTe = {:.1f} eV".
                            format(phys_numbers[randidx[x, y], 0] / 1e6,
                                   phys_numbers[randidx[x, y], 1],
                                   phys_numbers[randidx[x, y], 2]),
                            transform=axes[x, y].transAxes)

        fig.savefig(save_path + 'surrogate-compare-epoch-{}'.format(epoch))

    def __init__(self):
        pass
