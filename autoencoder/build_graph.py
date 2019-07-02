import tensorflow as tf

def deep_7(hyperparams):
    # TODO: implemnet learning rate schedule
    # TODO: implement dropout
    # dropout_rate = 0.5

    with tf.name_scope("data"):
        X = tf.placeholder(tf.float32, [None, hyperparams['n_inputs']], name="X")
        # for not running batch normalization at inference time
        training = tf.placeholder_with_default(False, shape=(), name="training")

    dense_layer = partial(tf.layers.dense,
                          kernel_initializer=tf.contrib.layers
                          .variance_scaling_initializer(seed=hyperparams['seed']),
                          kernel_regularizer=tf.contrib.layers
                          .l2_regularizer(hyperparams['scale']))
    batch_norm = partial(tf.layers.batch_normalization, training=training,
                         momentum=hyperparams['momentum'])

    size_l1 = 200
    size_l2 = 100
    size_l3 = 50
    size_h = 20

    with tf.name_scope("nn"):
        enc1 = dense_layer(X, size_l1, name="enc1")  # no activation specified = linear
        enc_b1 = tf.nn.elu(batch_norm(enc1))
        enc2 = dense_layer(enc_b1, size_l2, name="enc2")
        enc_b2 = tf.nn.elu(batch_norm(enc2))
        enc3 = dense_layer(enc_b2, size_l3, name="enc3")
        enc_b3 = tf.nn.elu(batch_norm(enc3))
        h_base = dense_layer(enc_b3, size_h, name="h_base")
        h_base_b = tf.nn.elu(batch_norm(h_base))
        dec1 = dense_layer(h_base_b, size_l3, name="dec1")  # TODO: maybe implement weight tying
        dec_b1 = tf.nn.elu(batch_norm(dec1))
        dec2 = dense_layer(dec_b1, size_l2, name="dec2")
        dec_b2 = tf.nn.elu(batch_norm(dec2))
        dec3 = dense_layer(dec_b2, size_l1, name="dec3")
        dec_b3 = tf.nn.elu(batch_norm(dec3))
        output_layer = dense_layer(dec_b3, 500, name="output_layer")
        output_b = tf.nn.elu(batch_norm(output_layer))
        output = tf.identity(output_b, name="output")

    with tf.name_scope("loss"):
        loss_base = tf.nn.l2_loss(X - output, name="loss_base")
        loss_reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_total = tf.add_n([loss_base] + loss_reg, name="loss_total")

    with tf.name_scope("train"):
        optimizer = tf.train.MomentumOptimizer(hyperparams['learning_rate'],
                                               hyperparams['momentum'], use_nesterov=True)
        training_op = optimizer.minimize(loss_total)

    return training_op, loss_total, X, training, output