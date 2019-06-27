from comet_ml import Experiment
import tensorflow as tf
import numpy as np
from functools import partial
from datetime import datetime
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/phil/Desktop/sweeps/sweep-langmuir-ml/data_preprocessing')
import preprocess


def build_graph(hyperparams):
    # TODO: implemnet learning rate schedule
    # TODO: implement dropout
    # dropout_rate = 0.5

    dense_layer = partial(tf.layers.dense, activation=tf.nn.relu,
                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(hyperparams['scale']))

    with tf.name_scope("data"):
        X = tf.placeholder(tf.float32, [None, hyperparams['n_inputs']], name="X")

    with tf.name_scope("nn"):
        enc1 = dense_layer(X, 200, name="enc1")
        enc2 = dense_layer(enc1, 100, name="enc2")
        enc3 = dense_layer(enc2, 50, name="enc3")
        h_base = dense_layer(enc3, 20, name="h_base")
        dec1 = dense_layer(h_base, 50, name="dec1")  # TODO: maybe implement weight tying
        dec2 = dense_layer(dec1, 100, name="dec2")
        dec3 = dense_layer(dec2, 200, name="dec3")
        output_layer = dense_layer(dec3, 500, name="output_layer")
        output = tf.identity(output_layer, name="output")

    with tf.name_scope("loss"):
        loss_base = tf.nn.l2_loss(X - output, name="loss_base")
        loss_reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_total = tf.add_n([loss_base] + loss_reg, name="loss_total")

    with tf.name_scope("train"):
        optimizer = tf.train.MomentumOptimizer(hyperparams['learning_rate'],
                                               hyperparams['momentum'], use_nesterov=True)
        training_op = optimizer.minimize(loss_total)

    return training_op, loss_total, X, output


# TODO: make plots of original and reconstructed traces
def plot_comparison(sess, data_train, data_test, X, output, hyperparams):
    output_train = output.eval(session=sess, feed_dict={X: data_train})
    output_test = output.eval(session=sess, feed_dict={X: data_test})

    fig, axes = plt.subplots(nrows=2, ncols=2)

    np.random.seed(hyperparams['seed'])
    randidx = np.random.randint(data_train.shape[0], size=2)
    axes[0, 0].plot(data_train[randidx[0]], label="Original")
    axes[0, 0].plot(output_train[randidx[0]], label="Reconstruction")
    axes[0, 1].plot(data_train[randidx[1]])
    axes[0, 1].plot(output_train[randidx[1]])
    axes[0, 0].set_ylabel('Training set', rotation=0)
    axes[0, 0].legend()

    np.random.seed(hyperparams['seed'])
    randidx = np.random.randint(data_test.shape[0], size=2)
    axes[1, 0].plot(data_test[randidx[0]])
    axes[1, 0].plot(output_test[randidx[0]])
    axes[1, 1].plot(data_test[randidx[1]])
    axes[1, 1].plot(output_test[randidx[1]])
    axes[1, 0].set_ylabel('Testing set', rotation=0)

    fig.tight_layout()

    return fig, axes


def train(hyperparams):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    experiment = Experiment(project_name="sweep-langmuir-ml", workspace="physicistphil")
    experiment.log_parameters(hyperparams)

    training_op, loss_total, X, output = build_graph(hyperparams)

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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        init.run()
        experiment.set_model_graph(sess.graph)

        for epoch in range(hyperparams['steps']):
            for i in range(data_train.shape[0] // hyperparams['batch_size']):
                X_batch = data_train[i * hyperparams['batch_size']:
                                     (i + 1) * hyperparams['batch_size']]
                sess.run(training_op, feed_dict={X: X_batch})
            X_batch = data_train[i * hyperparams['batch_size']:]
            sess.run(training_op, feed_dict={X: X_batch})

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


if __name__ == '__main__':
    hyperparams = {'n_inputs': 500,
                   'scale': 1e-2,
                   'learning_rate': 1e-3,
                   'momentum': 0.9,
                   'frac_train': 0.6,
                   'frac_test': 0.2,
                   'frac_valid': 0.2,
                   'batch_size': 64,
                   'steps': 1000,
                   'seed': 42}
    train(hyperparams)
