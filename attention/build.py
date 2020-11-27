import tensorflow as tf
from functools import partial
import numpy as np
from matplotlib import pyplot as plt


# Using a class allows us to access model tensors easily while maintaining some sense of order.
class Model:
    def build_data_pipeline(self, hyperparams, data_mean, data_ptp):
        with tf.compat.v1.variable_scope("pipeline"):
            # Preload both the train and test datasets so that we can switch between the two.
            self.data_train = tf.compat.v1.placeholder(tf.float32,
                                                       [None, hyperparams['n_inputs'] * 2 +
                                                        hyperparams['n_flag_inputs'] +
                                                        hyperparams['n_phys_inputs']])
            self.data_test = tf.compat.v1.placeholder(tf.float32,
                                                      [None, hyperparams['n_inputs'] * 2 +
                                                       hyperparams['n_flag_inputs'] +
                                                       hyperparams['n_phys_inputs']])

            # Keep mean and ptp of the data in the graph so they can be accessed outside
            #   of the model. We use this so that the error can be 
            self.data_mean = tf.constant(data_mean, dtype=np.float32, name="data_mean")
            self.data_ptp = tf.constant(data_ptp, dtype=np.float32, name="data_ptp")
            # self.dataset = tf.data.Dataset.from_tensor_slices(self.data_input_sliced)

            self.dataset_train = tf.data.Dataset.from_tensor_slices(self.data_train)
            self.dataset_train = self.dataset_train.batch(hyperparams['batch_size'])
            self.dataset_train = self.dataset_train.repeat()
            self.dataset_train = self.dataset_train.prefetch(tf.data.experimental.AUTOTUNE)
            self.data_train_iter = tf.compat.v1.data.make_initializable_iterator(self.dataset_train)

            self.dataset_test = tf.data.Dataset.from_tensor_slices(self.data_test)
            self.dataset_test = self.dataset_test.batch(hyperparams['batch_size'])
            self.dataset_test = self.dataset_test.repeat()
            self.dataset_test = self.dataset_test.prefetch(tf.data.experimental.AUTOTUNE)
            self.data_test_iter = tf.compat.v1.data.make_initializable_iterator(self.dataset_test)

            # This is a semi-supervised approach so splitting between features X and labels y
            #   doesn't really make much sense (real examples don't have labels, synthetic do).
            self.data_X_train = self.data_train_iter.get_next()
            self.data_X_test = self.data_test_iter.get_next()

    def build_X_switch(self, hyperparams):
        with tf.compat.v1.variable_scope("data"):
            self.training = tf.compat.v1.placeholder_with_default(False, shape=(), name="training")
            self.X = tf.cond(pred=self.training, true_fn=lambda: self.data_X_train,
                             false_fn=lambda: self.data_X_test)
            self.X_phys = tf.identity(self.X[:, hyperparams['n_inputs'] * 2:], name="X_phys")
            self.X = tf.identity(self.X[:, 0:hyperparams['n_inputs'] * 2], name="X")
            self.X_shape = tf.shape(input=self.X)

            # X needs to be 4d for input into convolution layers
            self.X_reshaped = tf.reshape(self.X, [-1, 2, hyperparams['n_inputs'], 1])

    def build_feature_extractor(self, hyperparams):
        conv_layer = partial(tf.compat.v1.layers.conv2d, activation=None,
                             kernel_initializer=tf.compat.v1.keras.initializers
                             .VarianceScaling(scale=2.0, seed=hyperparams['seed']),
                             kernel_regularizer=tf.keras.regularizers
                             .l2(0.5 * (hyperparams['l2_CNN'])),
                             )

        # batch_norm = partial(tf.compat.v1.layers.batch_normalization, training=self.training,
        #                      momentum=hyperparams['batch_momentum'], renorm=True)

        # This CNN extracts features from the input (stacked) voltage and current curves. We need
        #   to extract the features before applying the attention mask. If there was no feature
        #   extraction, each part of the sweep would be weighted, disorting the sweep. That's the
        #   idea at least -- I haven't verified that it actually works.
        feat_filters = hyperparams['feat_filters']
        with tf.compat.v1.variable_scope("feat"):
            # First layer looks at a portion of the vsweep, current. This portion is then passed
            #   into what are essentially feedforward networks (with width "filters" and
            #   parameter sharing) to transform the representation into something more abstract
            #   and amenable to attention weighting.
            self.feat_conv0 = conv_layer((self.X_reshaped),
                                         name="feat_conv0", filters=feat_filters,
                                         kernel_size=(2, 16), strides=(2, 1), padding='same',
                                         activation=tf.nn.elu)
            self.feat_conv1 = conv_layer((self.feat_conv0),
                                         name="feat_conv1", filters=feat_filters,
                                         kernel_size=(1, 1), strides=(1, 1), padding='same',
                                         activation=tf.nn.elu)
            self.feat_conv2 = conv_layer((self.feat_conv1),
                                         name="feat_conv2", filters=feat_filters,
                                         kernel_size=(1, 1), strides=(1, 1), padding='same',
                                         activation=tf.nn.elu)

    # CNN to build the attention mask. This chooses which portions of the sweep to focus on.
    #   The features extracted from the sweeps are multiplied by this mask and passed on to the
    #   translator. The error for each part of the sweep is also multiplied by this mask, so
    #   that portions that are not important / selected do not factor into the loss.
    def build_attention_focuser(self, hyperparams):
        conv_layer = partial(tf.compat.v1.layers.conv2d, activation=None,
                             kernel_initializer=tf.compat.v1.keras.initializers
                             .VarianceScaling(scale=2.0, seed=hyperparams['seed']),
                             kernel_regularizer=tf.keras.regularizers
                             .l2(0.5 * (hyperparams['l2_CNN'])),
                             )

        attn_filters = hyperparams['attn_filters']
        with tf.compat.v1.variable_scope("attn"):
            self.attn_conv0 = conv_layer((self.X_reshaped), name="attn_conv0",
                                         filters=attn_filters,
                                         kernel_size=(2, 32), strides=(2, 1), padding='same',
                                         activation=tf.nn.elu)
            self.attn_conv1 = conv_layer((self.attn_conv0), name="attn_conv1",
                                         filters=attn_filters,
                                         kernel_size=(1, 32), strides=(1, 1), padding='same',
                                         activation=tf.nn.elu)
            self.attn_conv2 = conv_layer((self.attn_conv1), name="attn_conv2",
                                         filters=attn_filters,
                                         kernel_size=(1, 32), strides=(1, 1), padding='same',
                                         activation=tf.nn.elu)
            self.attn_conv3 = conv_layer((self.attn_conv2), name="attn_conv3",
                                         filters=attn_filters,
                                         kernel_size=(1, 32), strides=(1, 1), padding='same',
                                         activation=tf.nn.elu)
            self.attn_conv4 = conv_layer((self.attn_conv3), name="attn_conv4",
                                         filters=attn_filters,
                                         kernel_size=(1, 32), strides=(1, 1), padding='same',
                                         activation=tf.nn.elu)
            self.attn_flat = conv_layer((self.attn_conv4), name="attn_flat", filters=1,
                                        kernel_size=(1, 1), strides=(1, 1), padding='valid')
            # This soft attention mask is shape (batch_size, 1, 256, 1)
            # I've found that sigmoid works better than softmax, which wasn't what I expected but
            #   I don't make the rules.
            self.attention_mask = tf.sigmoid((self.attn_flat))
            # Normalize the attention mask. As long as the scale of the mask is reasonable, I
            #   don't think it should affect training. Normalization is important so that all
            #   attention values cannot go to zero.
            self.attention_mask = tf.identity(self.attention_mask /
                                              tf.math.reduce_mean(self.attention_mask, axis=2,
                                                                  keepdims=True),
                                              name="attention_mask")
            # Mask the feature vector with the attention mask to make the glimpse.
            self.attention_glimpse = self.attention_mask * self.feat_conv2

    # Translate from the attention glimpse to physical parameters to calculate a theoretical
    #   Langmuir sweep. The attention glimpse is likely not in a useful / understandable
    #   representation, requiring a nonlinear transformation to physics space.
    def build_physics_translator(self, hyperparams):
        conv_layer = partial(tf.compat.v1.layers.conv2d, activation=None,
                             kernel_initializer=tf.compat.v1.keras.initializers
                             .VarianceScaling(scale=2.0, seed=hyperparams['seed']),
                             kernel_regularizer=tf.keras.regularizers
                             .l2(0.5 * (hyperparams['l2_CNN'])),
                             )
        pool_layer = partial(tf.compat.v1.layers.average_pooling2d, padding='same')
        dense_layer = partial(tf.compat.v1.layers.dense,
                              kernel_initializer=tf.compat.v1.keras.initializers
                              .VarianceScaling(scale=2.0, seed=hyperparams['seed']),
                              kernel_regularizer=tf.keras.regularizers
                              .l2(0.5 * (hyperparams['l2_translator'])))

        filters = hyperparams['filters']
        with tf.compat.v1.variable_scope("nn"):
            self.layer_conv0 = conv_layer((self.attention_glimpse), name="layer_conv0",
                                          filters=filters, kernel_size=(1, 8), strides=(1, 2),
                                          padding='same', activation=tf.nn.elu)
            self.layer_pool0 = pool_layer(self.layer_conv0, name="layer_pool0",
                                          pool_size=(1, 8), strides=(1, 2))

            self.layer_conv1 = conv_layer((self.layer_pool0), name="layer_conv1",
                                          filters=filters * 2, kernel_size=(1, 8), strides=(1, 2),
                                          padding='same', activation=tf.nn.elu)
            self.layer_pool1 = pool_layer(self.layer_conv1, name="layer_pool1",
                                          pool_size=(1, 8), strides=(1, 2))

            self.layer_conv2 = conv_layer((self.layer_pool1), name="layer_conv2",
                                          filters=filters * 4, kernel_size=(1, 8), strides=(1, 2),
                                          padding='same', activation=tf.nn.elu)
            self.layer_pool2 = pool_layer(self.layer_conv2, name="layer_pool2",
                                          pool_size=(1, 8), strides=(1, 2))

            middle_size = 4
            # Reshape for input into dense layers or whatever (TensorFlow needs explicit
            #   dimensions for NNs except for the batch size).
            self.conv_flattened = tf.reshape(self.layer_pool2, [-1, middle_size * filters * 4])
            self.layer_nn1 = dense_layer((self.conv_flattened), 32, activation=tf.nn.elu)
            self.layer_nn2 = dense_layer((self.layer_nn1), 32, activation=tf.nn.elu)
            self.layer_nn3 = dense_layer(self.layer_nn2, 32, activation=tf.nn.elu)
            self.CNN_output = tf.identity(self.layer_nn3, name='CNN_output')

        # Finish up the translation component with linear NN layers (not necessary, but common in
        #   ML models).
        # Also learn the vertical sweep offset which assumes zero ion saturation current.This
        #   may / should be removed in the future.
        n_phys_inputs = hyperparams['n_phys_inputs']
        with tf.compat.v1.variable_scope("trans"):
            # This is the learned offset for the sweep to be applied to theory curves.
            self.layer_offset = dense_layer(self.CNN_output, 1, name="layer_offset") / 1000.0
            # Divided by 1000 to make it easier to learn. We're learning small offsets and default
            #   values are much larger (this is essentially a somewhat manual initialization).

            # This gets passed off to the surrogate model
            self.phys_input = dense_layer(self.CNN_output, n_phys_inputs, name="phys_input")

    # Needed to put this in its own function because of needing to import the surrogate model
    #   meta graph after building the translator so we can get the scalefactor. Keeps things
    #   cleaner this way, hopefully.
    def build_plasma_info(self, scalefactor):
        # Divide by some constants to get physical numbers. Only take
        #   the first three components because the 4th one is for vsweep (and it's just 1.0).
        self.plasma_info = tf.identity(self.phys_input / scalefactor[0:3],
                                       # tf.concat([scalefactor[0:3], scalefactor[4:6]], 0),
                                       name="plasma_info")

    # Normalize the output of the surrogate / theory model so it can be directly compared with the
    #   normalized input data.
    def build_surrogate_output_normalizer(self, hyperparams, theory_output):
        # Scale the (physical) theory output to match that of the input curves (which *was* scaled)
        self.theory_output_normed = ((theory_output - self.data_mean[hyperparams['n_inputs']:]) /
                                     self.data_ptp[hyperparams['n_inputs']:])
        # Add the learned sweep offset
        self.theory_output_normed = tf.identity(self.theory_output_normed + self.layer_offset,
                                                name="theory_output_normed")

    # Construct the loss. The main components are: L2 error of model output to input sweeps only
    #   for attention-glimpsed regions (only for the current -- voltage is always given),
    #   L2 error of sweeps that have labels (plasma parameters) (i.e., synthetic sweeps),
    #   L1 of CNN output to, encourage sparsity (usually set to just 0), and regularization losses.
    def build_loss(self, hyperparams, scalefactor):
        with tf.compat.v1.variable_scope("loss"):
            self.X_I = self.X[:, hyperparams['n_inputs']:]
            # For backwards compatibility ._. (keeping tensor names the same)
            self.model_output = tf.identity(self.theory_output_normed, name="model_output")

            # L2 loss on the plasma parmaeters if they are given
            #   (set by the first X_phys ouput flag).
            self.loss_physics = (hyperparams['loss_physics'] * 0.5 *
                                 tf.reduce_sum(input_tensor=tf.expand_dims(self.X_phys[:, 0], 1) *
                                               (self.X_phys[:, 1:4] *
                                                scalefactor[0:3] - self.phys_input[:, 0:3]) ** 2))

            # L1 on CNN output for sparsity (I usally just set this to 0 but if you want to try...).
            self.l1_CNN_output = (hyperparams['l1_CNN_output'] *
                                  tf.reduce_sum(input_tensor=tf.math.abs(self.CNN_output)))

            # Penalize errors in the rebuilt current trace.
            self.loss_rebuilt = (tf.reduce_sum(input_tensor=(self.X_I - self.model_output) ** 2 *
                                               self.attention_mask[:, 0, :, 0],
                                               name="loss_rebuilt") *
                                 hyperparams['loss_rebuilt'])

            # Divide model loss by batch size to keep loss consistent regardless of input size.
            self.loss_model = ((self.loss_rebuilt +
                                # self.attention_loss +
                                self.loss_physics +
                                self.l1_CNN_output) /
                               tf.cast(tf.shape(input=self.model_output)[0], tf.float32))
            self.loss_reg = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
            self.loss_total = tf.add_n([self.loss_model] + self.loss_reg, name="loss_total")

        # I've had better luck with adam than anything else, but it's known to have worse
        #   generalization than tuned momentum gradient descent, I think.
        with tf.compat.v1.variable_scope("train"):
            # self.opt = tf.compat.v1.train.MomentumOptimizer(hyperparams['learning_rate'],
            #                                                 hyperparams['momentum'],
            #                                                 use_nesterov=True)
            self.opt = tf.compat.v1.train.AdamOptimizer(hyperparams['learning_rate'],
                                                        hyperparams['beta1'],
                                                        hyperparams['beta2'],
                                                        hyperparams['epsilon'])
            # If you want to try using tensor cores as part of the training process
            #   (but it's been slower or just NaNs for me for some reason).
            # self.opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(self.opt)
            self.grads = self.opt.compute_gradients(self.loss_total, var_list=self.vars)
            self.training_op = self.opt.apply_gradients(self.grads)

    # Load in a saved model. It needs to be exactly the same as the one constructed by the class.
    #   I have yet to incorporate a general loading mechanism.
    def load_model(self, sess, model_path):
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(sess, model_path)
        print("Model {} has been loaded.".format(model_path))

    # Plot a random subset of the testing set to guage fitting performance of the model. This is
    #   the main way I judge the performance of the model -- I've found that loss alone
    #   is insufficient.
    def plot_comparison(self, sess, hyperparams, save_path, epoch):
        (model_output, theory_output, phys_numbers, data_mean, data_ptp, data_input, attn_mask
         ) = sess.run([self.model_output, self.theory_output_normed, self.plasma_info,
                       self.data_mean[hyperparams['n_inputs']:],
                       self.data_ptp[hyperparams['n_inputs']:],
                       self.X,
                       self.attention_mask],
                      feed_dict={self.training: False})

        batch_size = theory_output.shape[0]

        # Input current.
        data_input = data_input[:, hyperparams['n_inputs']:] * data_ptp + data_mean

        # Output of the surrogate model with the learned plasma parameters.
        theory_output = theory_output * data_ptp + data_mean

        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8), sharex=True)
        fig.suptitle('Comparison of ')
        np.random.seed(hyperparams['seed'])
        randidx = np.random.randint(batch_size, size=(3, 4))

        # Plot the input and output current curves.
        for x, y in np.ndindex((3, 4)):
            axes[x, y].plot(data_input[randidx[x, y]], label="Data")
            axes[x, y].plot(theory_output[randidx[x, y]], label="Theory", alpha=0.8)
            axes[x, y].set_title("Index {}".format(randidx[x, y]))
        axes[0, 0].legend()

        for x, y in np.ndindex((3, 4)):
            # Print text with the plasma parameters on the plot.
            axes[x, y].text(0.05, 0.4,
                            "ne = {:3.1e} / cm$^3$ \nVp = {:.1f} V \nTe = {:.1f} eV".
                            format(phys_numbers[randidx[x, y], 0] / 1e6,
                                   phys_numbers[randidx[x, y], 1],
                                   phys_numbers[randidx[x, y], 2] / 1.602e-19),
                            transform=axes[x, y].transAxes,
                            fontsize=6)
            # Print the maximum attention value (useful for evaluating the attention portion).
            axes[x, y].text(0.6, 0.05,
                            "attn max: {:.3f}".format(np.max(attn_mask[randidx[x, y], 0, :, 0])),
                            transform=axes[x, y].transAxes, fontsize=6)
            # Display a heatmap of the attention values on the plot.
            mask_color = np.ones((attn_mask[randidx[x, y]].shape[1], 4))
            mask_color[:, 0] = 0.0
            mask_color[:, 2] = 0.0
            mask_color[:, 3] = (attn_mask[randidx[x, y], 0, :, 0] /
                                np.max(attn_mask[randidx[x, y], 0, :, 0]))
            mask_color = mask_color[np.newaxis, :, :]
            axes[x, y].imshow(mask_color, aspect='auto',
                              extent=(0.0, 256.0,
                                      axes[x, y].get_ylim()[0], axes[x, y].get_ylim()[1]))

        fig.savefig(save_path + 'full-compare-epoch-{}'.format(epoch))
        plt.close(fig)

    def __init__(self):
        pass
