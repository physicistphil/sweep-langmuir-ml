from comet_ml import Experiment
from comet_ml import Optimizer

import tensorflow as tf
import numpy as np
from functools import partial
from datetime import datetime

# custom tools
import sys
sys.path.append('/home/phil/Desktop/sweeps/sweep-langmuir-ml/data_preprocessing')
import preprocess
import plot_utils


def build_graph(hyperparams):
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

    with tf.name_scope("nn"):
        enc1 = dense_layer(X, 200, name="enc1")
        enc_b1 = tf.nn.elu(batch_norm(enc1))
        enc2 = dense_layer(enc_b1, 100, name="enc2")
        enc_b2 = tf.nn.elu(batch_norm(enc2))
        enc3 = dense_layer(enc_b2, 50, name="enc3")
        enc_b3 = tf.nn.elu(batch_norm(enc3))
        h_base = dense_layer(enc_b3, 20, name="h_base")
        h_base_b = tf.nn.elu(batch_norm(h_base))
        dec1 = dense_layer(h_base_b, 50, name="dec1")  # TODO: maybe implement weight tying
        dec_b1 = tf.nn.elu(batch_norm(dec1))
        dec2 = dense_layer(dec_b1, 100, name="dec2")
        dec_b2 = tf.nn.elu(batch_norm(dec2))
        dec3 = dense_layer(dec_b2, 200, name="dec3")
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


def train(experiment, hyperparams):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    training_op, loss_total, X, training, output = build_graph(hyperparams)

    np.random.seed(hyperparams['seed'])
    data = preprocess.get_mirror_data(hyperparams['n_inputs'])
    np.random.shuffle(data)
    data_size = data.shape[0]
    data_train = data[0:int(data_size * hyperparams['frac_train']), :]
    data_test = data[int(data_size * hyperparams['frac_test']):
                     int(data_size * (hyperparams['frac_test'] + hyperparams['frac_valid'])), :]
    data_valid = data[int(data_size * hyperparams['frac_valid']):, :]

    experiment.log_dataset_hash(data)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # for batch normalization updates
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init.run()
        experiment.set_model_graph(sess.graph)

        for epoch in range(hyperparams['steps']):
            for i in range(data_train.shape[0] // hyperparams['batch_size']):
                X_batch = data_train[i * hyperparams['batch_size']:
                                     (i + 1) * hyperparams['batch_size']]
                sess.run([training_op, extra_update_ops], feed_dict={X: X_batch, training: True})
            X_batch = data_train[(i + 1) * hyperparams['batch_size']:]
            sess.run([training_op, extra_update_ops], feed_dict={X: X_batch, training: True})

            if epoch % 10 == 0:
                loss_train = loss_total.eval(feed_dict={X: data_train}) / data_train.shape[0]
                loss_test = loss_total.eval(feed_dict={X: data_test}) / data_test.shape[0]
                with experiment.train():
                    experiment.log_metric('Loss', loss_train, step=epoch)
                with experiment.test():
                    experiment.log_metric('Loss', loss_test, step=epoch)
                print("Epoch {:5}\tWall: {} \tTraining loss: {:.4e}\tTesting loss: {:.4e}"
                      .format(epoch, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                              loss_train, loss_test))

            if epoch % 100 == 0:
                saver.save(sess, "./saved_models/ae-001-{}.ckpt".format(now))

        loss_train = loss_total.eval(feed_dict={X: data_train}) / data_train.shape[0]
        loss_test = loss_total.eval(feed_dict={X: data_test}) / data_test.shape[0]
        print("Epoch {:5}\tWall: {} \tTraining loss: {:.4e}\tTesting loss: {:.4e}"
              .format(epoch, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                      loss_train, loss_test))
        with experiment.train():
            experiment.log_metric('Loss', loss_train)
        with experiment.test():
            experiment.log_metric('Loss', loss_test)
        saver.save(sess, "./saved_models/ae-001-{}-final.ckpt".format(now))

        experiment.set_step(epoch)
        fig_compare, axes = plot_utils.plot_comparison(sess, data_test, X, output, hyperparams)
        fig_worst, axes = plot_utils.plot_worst(sess, data_train, X, output, hyperparams)
        experiment.log_figure(figure_name="comparison", figure=fig_compare)
        experiment.log_figure(figure_name="worst examples", figure=fig_worst)
        experiment.log_asset_data


if __name__ == '__main__':
    experiment = Experiment(project_name="sweep-langmuir-ml", workspace="physicistphil")

    opt_config = {"algorithm": "bayes",
                  "parameters": {"scale": {"type": "float",
                                           "scalingType": "loguniform",
                                           "min": 0.0
                                            },
                                 "learning_rate": {},
                                 "momentum": {}},
                  "spec": {"metric": "loss",
                           "objective": "minimize"}}

    hyperparams = {'n_inputs': 500,
                   'scale': 0.0,  # no regularization
                   'learning_rate': 1e-3,
                   'momentum': 0.9,
                   'frac_train': 0.6,
                   'frac_test': 0.2,
                   'frac_valid': 0.2,
                   'batch_size': 128,
                   'steps': 100,
                   'seed': 42}

    experiment.log_parameters(hyperparams)

    train(hyperparams)
