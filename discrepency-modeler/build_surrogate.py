import tensorflow as tf
from functools import partial


class DenseNN:
    def build_data_pipeline(self, hyperparams, generator):
        with tf.name_scope("pipeline"):
            input_size = hyperparams['n_phys_inputs'] + hyperparams['n_inputs']
            output_size = hyperparams['n_output']
            self.dataset = tf.data.Dataset.from_generator(lambda: generator(hyperparams, limit=-1),
                                                          (tf.float32, tf.float32),
                                                          ([None, input_size], [None, output_size]))
            # Batch size is not 'batch_size' because the output from the generator is already
            #   the size of one batch.
            self.dataset = self.dataset.batch(1)
            self.dataset = self.dataset.prefetch(2)  # Prefetch 2 batches

            self.data_iter = self.dataset.make_initializable_iterator()
            self.data_input, self.data_output = self.data_iter.get_next()

    # Pass X and y as parameters so that creation of the dataset or input is guaranteed to precede
    #   NN graph construction. Building the main graph separately enables easier loading of
    #   pretrained models.
    def build_NN(self, hyperparams, X, y):
        with tf.name_scope("data"):
            self.training = tf.compat.v1.placeholder_with_default(False, shape=(), name="training")
            # self.X_ptp = tf.compat.v1.placeholder(tf.float32, name="X_ptp")
            # self.X_mean = tf.compat.v1.placeholder(tf.float32, name="X_mean")

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
        batch_norm = partial(tf.layers.batch_normalization, training=self.training,
                             momentum=hyperparams['momentum'])

        size_l1 = hyperparams['size_l1']
        size_l2 = hyperparams['size_l2']
        n_output = hyperparams['n_output']

        with tf.name_scope("nn"):

            self.nn_layer1 = dense_layer(X, size_l1, name="nn_layer1")
            self.nn_activ1 = tf.nn.elu(batch_norm(self.nn_layer1), name="nn_activ1")

            self.nn_layer2 = dense_layer(self.nn_activ1, size_l2, name="nn_layer2")
            self.nn_activ2 = batch_norm(self.nn_layer2, name="nn_activ2")

            self.nn_layer_out = dense_layer(self.nn_activ2, n_output, name="nn_layer_out")
            self.nn_activ_out = tf.nn.elu(batch_norm(self.nn_layer_out), name="nn_activ_out")

            self.nn_output = tf.identity(self.nn_activ_out, name="output")

        with tf.variable_scope("loss"):
            self.loss_base = tf.nn.l2_loss(self.nn_output - y, name="loss_base")
            self.loss_reg = tf.compat.v1.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss_total = tf.add_n([self.loss_base] + self.loss_reg, name="loss_total")

        with tf.variable_scope("trainer"):
            self.optimizer = tf.compat.v1.train.MomentumOptimizer(hyperparams['learning_rate'],
                                                                  hyperparams['momentum'],
                                                                  use_nesterov=True)

            self.grads = self.optimizer.compute_gradients(self.loss_total)
            self.training_op = self.optimizer.apply_gradients(self.grads)

    def load_model(self, model_path):
        pass

    # Nothing is built by default so that we can build graph components for use
    #   in different models.
    def __init__(self):
        pass
