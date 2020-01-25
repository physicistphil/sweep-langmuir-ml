import tensorflow as tf
from functools import partial
import numpy as np
from matplotlib import pyplot as plt


class Model:
    def build_data_pipeline(self, hyperparams, generator):
        with tf.name_scope("pipeline"):
            input_size = hyperparams['n_phys_inputs'] + hyperparams['n_inputs']
            output_size = hyperparams['n_output']
            self.dataset = tf.data.Dataset.from_generator(lambda: generator(hyperparams, limit=-1),
                                                          (tf.float32, tf.float32),
                                                          ([None, input_size], [None, output_size]))
            # Batch size is not 'batch_size' because the output from the generator is already
            #   the size of one batch: taking one element from the dataset is actually a batch.
            self.dataset = self.dataset.batch(1)
            self.dataset = self.dataset.prefetch(12)  # Prefetch 2 batches

            self.data_iter = self.dataset.make_initializable_iterator()
            self.data_X, self.data_y = self.data_iter.get_next()

    # Pass X and y as parameters so that creation of the dataset or input is guaranteed to precede
    #   NN graph construction. Building the main graph separately enables easier loading of
    #   pretrained models.
    def build_dense_NN(self, hyperparams, X, y):
        # Training boolean placeholder is necessary for batch normalization.
        self.training = tf.compat.v1.placeholder_with_default(False, shape=(), name="training")
        dense_layer = partial(tf.layers.dense, kernel_initializer=tf.contrib.layers
                              .variance_scaling_initializer(seed=hyperparams['seed']),
                              kernel_regularizer=tf.contrib.layers
                              .l2_regularizer(hyperparams['l2_scale']))
        batch_norm = partial(tf.layers.batch_normalization, training=self.training,
                             momentum=hyperparams['batch_momentum'])

        with tf.name_scope("data"):
            # We need to train the surrogate on each indiviudal sweep point instead of the whole
            #   sweep at once -- we want to learn f(voltage), not f(entire voltage sweep). This
            #   should simplify the model and make it easier to train.
            n_phys_inputs = hyperparams['n_phys_inputs']
            n_inputs = hyperparams['n_inputs']
            X = tf.squeeze(X)  # Get rid of excess dimensions.
            self.X_phys_repeated = tf.expand_dims(X[:, 0:n_phys_inputs], 1)
            # Size of physical parameters tensor is now [batch_size, 1, n_phys_input].
            # Repeat ne, Vp, Te over axis 1.
            self.X_phys_repeated = tf.tile(self.X_phys_repeated, [1, n_inputs, 1])
            # Size is now [batch_size, n_inputs, n_phys_inputs].
            self.X_sweep = tf.expand_dims(X[:, n_phys_inputs:], 2)
            # Size of sweep tensor is now [batch_size, n_inputs, 1].
            # Stack the physical parameters tensor and vsweep tensor together.
            self.X_repeated = tf.concat([self.X_phys_repeated, self.X_sweep], 2)
            # Size is now [batch_size, n_inputs, n_phys_inputs + 1].
            X = self.X_repeated
            # Our network will now be fed tensors of [batch_size * n_inputs, 4] and will output
            #   and will output tensors of [batch_size * n_inputs, 1].
            X = (tf.reshape(X, [-1, n_phys_inputs + 1]))
            # Rescale so training is easier.
            self.X_scalefactor = tf.constant([1e-17, 1e-1, 1e19, 1e-1])
            X = X * self.X_scalefactor
            self.X = X

            # Match dimensions of y with X. Shape will be [batch_size, n_inputs, 1].
            y = (tf.reshape(y, [-1, 1]))
            # Rescale for easier training.
            self.y_scalefactor = tf.constant([1e2])
            y = y * self.y_scalefactor
            self.y = y

        size_l1 = hyperparams['size_l1']
        size_l2 = hyperparams['size_l2']
        # n_output = hyperparams['n_output']
        n_output = 1

        with tf.name_scope("nn"):
            self.nn_layer1 = dense_layer(X, size_l1, name="nn_layer1")
            self.nn_activ1 = tf.nn.elu(batch_norm(self.nn_layer1), name="nn_activ1")

            self.nn_layer2 = dense_layer(self.nn_activ1, size_l2, name="nn_layer2")
            self.nn_activ2 = tf.nn.elu(batch_norm(self.nn_layer2, name="nn_activ2"))

            self.nn_layer3 = dense_layer(self.nn_activ2, size_l2, name="nn_layer3")
            self.nn_activ3 = tf.nn.elu(batch_norm(self.nn_layer3, name="nn_activ3"))

            self.nn_layer4 = dense_layer(self.nn_activ3, size_l2, name="nn_layer4")
            self.nn_activ4 = tf.nn.elu(batch_norm(self.nn_layer4, name="nn_activ4"))

            self.nn_layer_out = dense_layer(self.nn_activ4, n_output, name="nn_layer_out")
            self.nn_activ_out = tf.identity(batch_norm(self.nn_layer_out), name="nn_activ_out")

            # The output of the network should be a complete sweep (which is of length n_inputs)
            self.output = tf.reshape(self.nn_activ_out, [-1, n_inputs], name="output")

        with tf.variable_scope("loss"):
            self.loss_base = tf.nn.l2_loss(self.nn_activ_out * self.y_scalefactor - y,
                                           name="loss_base") / hyperparams['batch_size']
            self.loss_reg = tf.compat.v1.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss_total = tf.add_n([self.loss_base] + self.loss_reg, name="loss_total")

        with tf.variable_scope("trainer"):
            self.optimizer = tf.compat.v1.train.MomentumOptimizer(hyperparams['learning_rate'],
                                                                  hyperparams['momentum'],
                                                                  use_nesterov=True)

            self.grads = self.optimizer.compute_gradients(self.loss_total)
            self.training_op = self.optimizer.apply_gradients(self.grads)

    # Load the neural network model to be used as a step in another model.
    def load_model(self, model_path):
        pass

    # A quick diagnostic function to make sure the tensor shapes are correct.
    def check_sizes(self, sess):
        X_phys_repeated, X_sweep, X_repeated, X, y = sess.run([self.X_phys_repeated,
                                                               self.X_sweep, self.X_repeated,
                                                               self.X, self.y])
        print("X_phys_repeated.shape: {}\n".format(X_phys_repeated.shape) +
              "X_sweep.shape: {}\n".format(X_sweep.shape) +
              "X_repeated.shape: {}\n".format(X_repeated.shape) +
              "X.shape: {}\n".format(X.shape) +
              "y.shape: {}".format(y.shape))

        # np.set_printoptions(threshold=np.inf)
        print("X: {}".format(X))
        print("y: {}".format(y))

    # Plot results of analytic and surrogate models
    def plot_comparison(self, sess, hyperparams, save_path, epoch):
        # print("Plotting comparison")

        # Shape of data_y without squeeze is [1, batch_size, n_inputs]
        # Shape of nn_output is [batch_size, n_inputs]
        data_y, model_output = sess.run([self.data_y, self.output],
                                        feed_dict={self.training: False})
        data_y = np.squeeze(data_y)

        # generated_trace = output_test
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8), sharex=True)
        fig.suptitle('Comparison of ')
        np.random.seed(hyperparams['seed'])
        randidx = np.random.randint(data_y.shape[1], size=(3, 4))

        for x, y in np.ndindex((3, 4)):
            axes[x, y].plot(data_y[randidx[x, y]], label="Analytic")
            axes[x, y].plot(model_output[randidx[x, y]], label="Surrogate")
            axes[x, y].set_title("Index {}".format(randidx[x, y]))
        axes[0, 0].legend()
        fig.savefig(save_path + 'surrogate-compare-epoch-{}'.format(epoch))

        # print("Finished plotting comparison")

    # Nothing is built by default so that we can build graph components for use
    #   in different models.
    def __init__(self):
        pass
