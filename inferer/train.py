# This file should be run from the folder its in. E.g.: python train.py

import tensorflow as tf
import numpy as np
from datetime import datetime

# Modify log levels to keep console clean.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Custom tools from other directories
import sys
sys.path.append('/home/phil/Desktop/sweeps/sweep-langmuir-ml/data_processor')
import preprocess
sys.path.append('/home/phil/Desktop/sweeps/sweep-langmuir-ml/simulator')
import generate
sys.path.append('/home/phil/Desktop/sweeps/sweep-langmuir-ml/utilities')
import plot_utils

# From the autoencoder directory
import build_graph

# weights and biases -- ML experiment tracker
import wandb

# Get data from real experiments (so far just from the mirror dataset).
"""
    print("Getting data...", end=" ")
    sys.stdout.flush()
    data = preprocess.get_mirror_data(hyperparams['n_inputs'], return_grids=True)
    data_train, data_test, data_valid = preprocess.shuffle_split_data(data, hyperparams)
    experiment.log_dataset_hash(data_train)
    print("Done.")

    return data_train, data_test, data_valid
"""


# Generate synthetic sweeps and make conventional train / test / validation sets.
# Order is: ne (electron density), Vp (plasma potential), and Te (electron temperature)
def gather_synthetic_data(hyperparams):
    print("Generating traces...", end=" ")
    ne = np.linspace(1e17, 1e18, 20)  # Densities in m^-3 (typcial of LAPD)
    Vp = np.linspace(20, 60, 20)  # Plasma potential in V (typcial of LAPD? idk)
    e = 1.602e-19  # elemntary charge
    Te = np.linspace(0.5, 5, 40) * e  # Electron temperature in J (typical of LAPD)
    vsweep = np.linspace(-30, 70, hyperparams['n_inputs'])  # Sweep voltage in V
    S = 2e-6  # Probe area in m^2

    ne_grid, Vp_grid, Te_grid, vsweep_grid, data \
        = generate.generate_basic_trace(ne, Vp, Te, vsweep, S=S, return_grids=True)

    n_inputs = hyperparams['n_inputs']
    y = np.stack((ne_grid.reshape((-1, n_inputs)), Vp_grid.reshape((-1, n_inputs)),
                  Te_grid.reshape(-1, n_inputs)))[:, :, 0].transpose()
    y_train, y_test, y_valid = preprocess.shuffle_split_data(y, hyperparams)

    data = np.reshape(data, (-1, n_inputs))
    data_train, data_test, data_valid = preprocess.shuffle_split_data(data, hyperparams)
    # experiment.log_dataset_hash([data_train, y_train])
    print("Done.")

    # TODO: return vsweep as well to allow for training on different voltage sweeps
    return data_train, data_test, data_valid, y_train, y_test, y_valid


# Mix the generated synthetic sweeps and real data and return the combined set.
# We have 3264 real traces, so we probably want a similar amount of synthetic ones if we want to
# train them on equal footing.
def gather_mixed_data(experiment, hyperparams):
    pass


def train(hyperparams, debug=False):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    # choose which data to train on
    data_train, data_test, data_valid, y_train, y_test, y_valid \
        = gather_synthetic_data(hyperparams)

    if not debug:
        training_op, X, y, training, output, loss_total \
            = build_graph.make_small_nn(hyperparams, size_output=3)
    else:
        training_op, X, y, training, output, loss_total, grads \
            = build_graph.make_small_nn(hyperparams, size_output=3, debug=debug)
        for grad, var in grads:
            tf.compat.v1.summary.histogram("gradients/" + var.name, grad)
            tf.compat.v1.summary.histogram("variables/" + var.name, var)

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    # for batch normalization updates
    extra_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    # ---------------------- Begin training ---------------------- #
    with tf.compat.v1.Session(config=config) as sess:
        init.run()
        summaries_op = tf.compat.v1.summary.merge_all()
        summary_writer = tf.compat.v1.summary.FileWriter("summaries/sum-" + now, graph=sess.graph)

        for epoch in range(hyperparams['steps']):
            for i in range(data_train.shape[0] // hyperparams['batch_size']):
                X_batch = data_train[i * hyperparams['batch_size']:
                                     (i + 1) * hyperparams['batch_size']]
                y_batch = y_train[i * hyperparams['batch_size']:
                                  (i + 1) * hyperparams['batch_size']]
                sess.run([training_op, extra_update_ops],
                         feed_dict={X: X_batch, y: y_batch, training: True})
                if i == 0 and epoch % 10 == 0 and debug:
                    summary = sess.run(summaries_op,
                                       feed_dict={X: data_train, y: y_train, training: True})
                    summary_writer.add_summary(summary, epoch)
                    # maybe do over just a batch?
            X_batch = data_train[(i + 1) * hyperparams['batch_size']:]
            y_batch = y_train[(i + 1) * hyperparams['batch_size']:]
            sess.run([training_op, extra_update_ops],
                     feed_dict={X: X_batch, y: y_batch, training: True})

            print("[" + "=" * int(25.0 * (epoch % 10) / 10.0) +
                  " " * int(np.ceil(25.0 * (1.0 - (epoch % 10) / 10.0))) +
                  "]", end="")
            print("\r", end="")

            if epoch % 10 == 0:
                print("[" + "=" * 25 + "]", end="\t")

                loss_train = (loss_total.eval(feed_dict={X: data_train, y: y_train}) /
                              data_test.shape[0])
                loss_test = (loss_total.eval(feed_dict={X: data_test, y: y_test}) /
                             data_test.shape[0])
                wandb.log({'loss_train': loss_train, 'loss_test': loss_test}, step=epoch)
                print("Epoch {:5}\tWall: {} \tTraining: {:.4e}\tTesting: {:.4e}"
                      .format(epoch, datetime.utcnow().strftime("%H:%M:%S"),
                              loss_train, loss_test))
            if epoch % 100 == 0:
                saver.save(sess, "./saved_models/autoencoder-{}.ckpt".format(now))

        print("[" + "=" * 25 + "]", end="\t")

        # ---------------------- Log results ---------------------- #
        # calculate loss
        loss_train = loss_total.eval(feed_dict={X: data_train, y: y_train}) / data_train.shape[0]
        loss_test = loss_total.eval(feed_dict={X: data_test, y: y_test}) / data_test.shape[0]
        print("Epoch {:5}\tWall: {} \tTraining: {:.4e}\tTesting: {:.4e}"
              .format(epoch, datetime.utcnow().strftime("%H:%M:%S"),
                      loss_train, loss_test))
        wandb.log({'loss_train': loss_train, 'loss_test': loss_test}, step=epoch)
        saver.save(sess, "./saved_models/autoencoder-{}-final.ckpt".format(now))

        # calculate RMS percent error of quantities. quants output order is: ne, Vp, Te
        quants = output.eval(feed_dict={X: data_test, y: y_test})
        per_ne = np.std((quants[:, 0] - y_test[:, 0]) / y_test[:, 0] * 100 - 100)
        per_Vp = np.std((quants[:, 1] - y_test[:, 1]) / y_test[:, 1] * 100 - 100)
        per_Te = np.std((quants[:, 2] - y_test[:, 2]) / y_test[:, 2] * 100 - 100)
        print("RMS: \tne: {:03.1f}%\tVp: {:03.1f}%\tTe: {:03.1f}%\t".format(per_ne, per_Vp, per_Te))
        wandb.log({'RMS_pct_ne': per_ne, 'RMS_pct_Vp': per_Vp, 'RMS_pct_Te': per_Te}, step=epoch)

        # make plots comparing learned parameters to the actual ones
        fig_compare, axes = plot_utils.inferer_plot_comparison(sess, data_test, y_test, X, y,
                                                               output, hyperparams)
        # fig_worst, axes = plot_utils.plot_worst(sess, data_train, X, output, hyperparams)
        wandb.log({"comaprison_plot": fig_compare}, step=epoch)
        os.mkdir("plots/fig-{}".format(now))
        fig_compare.savefig("plots/fig-{}/compare".format(now))
        # fig_worst.savefig("plots/fig-{}/worst".format(now))

        # log tensorflow graph and variables
        checkpoint_name = "./saved_models/autoencoder-{}-final.ckpt".format(now)
        wandb.save(checkpoint_name + ".index")
        wandb.save(checkpoint_name + ".meta")
        wandb.save(checkpoint_name + ".data-00000-of-00001")


if __name__ == '__main__':
    hyperparams = {'n_inputs': 500,
                   'scale': 0.5,
                   'learning_rate': 1e-4,
                   'momentum': 0.9,
                   'frac_train': 0.6,
                   'frac_test': 0.2,
                   'frac_valid': 0.2,
                   'batch_size': 512,
                   'steps': 200,
                   'seed': 42}
    wandb.init(project="sweep-langmuir-ml", sync_tensorboard=True, config=hyperparams)
    train(hyperparams, debug=True)
