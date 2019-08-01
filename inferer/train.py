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

# From the inferer directory
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
def gather_synthetic_scaled_data(hyperparams):
    print("Generating traces...", end=" ")
    ne = np.linspace(1e17, 1e18, 20)  # Densities in m^-3 (typical of LAPD)
    Vp = np.linspace(20, 60, 20)  # Plasma potential in V (typical of LAPD? idk)
    e = 1.602e-19  # elementary charge
    Te = np.linspace(0.5, 5, 40) * e  # Electron temperature in J (typical of LAPD)
    vsweep = np.linspace(-30, 70, hyperparams['n_inputs'])  # Sweep voltage in V
    S = 2e-6  # Probe area in m^2

    ne_grid, Vp_grid, Te_grid, vsweep_grid, data \
        = generate.generate_basic_trace_from_grid(ne, Vp, Te, vsweep, S=S, return_grids=True)

    n_inputs = hyperparams['n_inputs']
    y = np.stack((ne_grid.reshape((-1, n_inputs)), Vp_grid.reshape((-1, n_inputs)),
                  Te_grid.reshape(-1, n_inputs)))[:, :, 0].transpose()
    mean = np.mean(y, axis=0)
    diff = np.max(y, axis=0) - np.min(y, axis=0)
    y = (y - mean) / diff
    y_train, y_test, y_valid = preprocess.shuffle_split_data(y, hyperparams)

    data = np.reshape(data, (-1, n_inputs))
    # Add random displacements to force vertical translation invariance in our inference.
    # This is to avoid digitizer offsets from messing with our estimates.
    # 0.1 Amp (~1 V) offset is typical.
    np.random.seed(hyperparams['seed'])
    data += np.random.uniform(-0.1, 0.1, size=(data.shape[0], 1))
    data_train, data_test, data_valid = preprocess.shuffle_split_data(data, hyperparams)
    print("Done.")

    # TODO: return vsweep as well to allow for training on different voltage sweeps
    return data_train, data_test, data_valid, y_train, y_test, y_valid, mean, diff


def gather_random_synthetic_scaled_data(hyperparams):
    size = 3072  # Number of examples to generate. This is comparable to the number of real sweeps.

    ne_range = np.array([1e17, 1e19])
    Vp_range = np.array([5, 15])
    e = 1.602e-19  # Elementary charge
    Te_range = np.array([1, 5]) * e  # We're defining it in terms of eV because it's comfortable.
    S = 2e-6  # Probe area in m^2

    # Voltages used when collecting real sweeps are within this range.
    vsweep_lower_range = np.array([-50, -20])
    vsweep_upper_range = np.array([50, 100])

    ne, Vp, Te, vsweep, current \
        = generate.generate_random_traces_from_array(ne_range, Vp_range, Te_range,
                                                     vsweep_lower_range, vsweep_upper_range,
                                                     hyperparams, size, S=S)

    # Find the voltage sweep and current means and peak-to-peaks so the model is easier to train.
    vsweep_mean = np.full(hyperparams['n_inputs'], np.mean(vsweep))
    vsweep_ptp = np.full(hyperparams['n_inputs'], np.ptp(vsweep))
    current_mean = np.full(hyperparams['n_inputs'], np.mean(current))
    current_ptp = np.full(hyperparams['n_inputs'], np.ptp(current))
    # Combine the two so we have a nice neat X, y, and scalings tuple returned by the function.
    X_mean = np.concatenate((vsweep_mean, current_mean))
    X_ptp = np.concatenate((vsweep_ptp, current_ptp))

    # Merge the voltage sweep and current into one array.
    # Because the network is fully connected, order does not matter.
    # This will probably need to changed if more complicated / powerful architectures are used.
    # We're also centering and scaling the input so that it's easier to train.
    X = (np.concatenate((vsweep, current), axis=1) - X_mean) / X_ptp
    X_train, X_test, X_valid = preprocess.shuffle_split_data(X, hyperparams)

    # Center and rescale the sweep parameters, otherwise the model is impossible to train.
    y = np.stack((ne, Vp, Te)).transpose()
    y_mean = np.mean(y, axis=0)
    y_ptp = np.ptp(y, axis=0)
    y = (y - y_mean) / y_ptp
    y_train, y_test, y_valid = preprocess.shuffle_split_data(y, hyperparams)

    return X_train, X_test, X_valid, X_mean, X_ptp, y_train, y_test, y_valid, y_mean, y_ptp


# Mix the generated synthetic sweeps and real data and return the combined set.
# We have 3264 real traces, so we probably want a similar amount of synthetic ones if we want to
#   train them on equal footing.
def gather_mixed_data(experiment, hyperparams):
    pass


def train(hyperparams, debug=False):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    wandb.log({"now": now}, step=0)
    os.mkdir("plots/fig-{}".format(now))

    # choose which data to train on
    # X_train, X_test, X_valid, y_train, y_test, y_valid, y_mean, y_diff \
    #   = gather_synthetic_scaled_data(hyperparams)
    X_train, X_test, X_valid, X_mean, X_ptp, y_train, y_test, y_valid, y_mean, y_ptp \
        = gather_random_synthetic_scaled_data(hyperparams)

    # Build the models to train on.
    if not debug:
        training_op, X, y, training, output, loss_total \
            = build_graph.make_small_nn(hyperparams, size_output=3)
    else:
        training_op, X, y, training, output, loss_total, grads \
            = build_graph.make_small_nn(hyperparams, size_output=3, debug=debug)
        for grad, var in grads:
            tf.compat.v1.summary.histogram("gradients/" + var.name, grad)
            tf.compat.v1.summary.histogram("variables/" + var.name, var)

    # Initialize configuration and variables.
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
        best_loss = 1000

        # Augment our test data.
        X_test = preprocess.add_offset_to_half(X_test, hyperparams, epoch=0)
        X_test = preprocess.add_noise_to_half(X_test, hyperparams, epoch=0)

        # X_train_aug = X_train
        for epoch in range(hyperparams['steps']):
            if epoch % 100 == 0:
                # Augment data each epoch.
                # Apply random offset to learn invariance.
                X_train_aug = preprocess.add_offset_to_half(X_train, hyperparams, epoch=epoch)
                # Apply noise.
                X_train_aug = preprocess.add_noise_to_half(X_train_aug, hyperparams, epoch=epoch)

            for i in range(X_train_aug.shape[0] // hyperparams['batch_size']):
                X_batch = X_train_aug[i * hyperparams['batch_size']:
                                      (i + 1) * hyperparams['batch_size']]
                y_batch = y_train[i * hyperparams['batch_size']:
                                  (i + 1) * hyperparams['batch_size']]
                sess.run([training_op, extra_update_ops],
                         feed_dict={X: X_batch, y: y_batch, training: True})
                if i == 0 and epoch % 10 == 0 and debug:
                    summary = sess.run(summaries_op,
                                       feed_dict={X: X_train_aug, y: y_train, training: True})
                    summary_writer.add_summary(summary, epoch)
            if (X_train_aug.shape[0] % hyperparams['batch_size']) != 0:
                X_batch = X_train_aug[(i + 1) * hyperparams['batch_size']:]
                y_batch = y_train[(i + 1) * hyperparams['batch_size']:]
                sess.run([training_op, extra_update_ops],
                         feed_dict={X: X_batch, y: y_batch, training: True})

            print("[" + "=" * int(25.0 * (epoch % 10) / 10.0) +
                  " " * int(np.ceil(25.0 * (1.0 - (epoch % 10) / 10.0))) +
                  "]", end="")
            print("\r", end="")

            # loss_test = (loss_total.eval(feed_dict={X: X_test, y: y_test}) /
            #              X_test.shape[0])
            # print("loss: {}, epoch: {}".format(loss_test, epoch))

            if epoch % 10 == 0:
                print("[" + "=" * 25 + "]", end="\t")

                loss_train = (loss_total.eval(feed_dict={X: X_train_aug, y: y_train}) /
                              X_train_aug.shape[0])
                loss_test = (loss_total.eval(feed_dict={X: X_test, y: y_test}) /
                             X_test.shape[0])
                wandb.log({'loss_train': loss_train, 'loss_test': loss_test}, step=epoch)
                print("Epoch {:5}\tWall: {} \tTraining: {:.4e}\tTesting: {:.4e}"
                      .format(epoch, datetime.utcnow().strftime("%H:%M:%S"),
                              loss_train, loss_test))

                if loss_test < best_loss:
                    best_loss = loss_test
                    saver.save(sess, "./saved_models/model-{}-best.ckpt".format(now))

            if epoch % 100 == 0:
                saver.save(sess, "./saved_models/model-{}.ckpt".format(now))
                # Make plots comparing learned parameters to the actual ones.
            if epoch % 1000 == 0:
                fig_compare, axes = plot_utils. \
                    inferer_plot_comparison_including_vsweep(sess, X, X_test, X_mean, X_ptp, output,
                                                             y_mean, y_ptp, hyperparams)
                # wandb.log({"comaprison_plot": fig_compare}, step=epoch)
                fig_compare.savefig("plots/fig-{}/compare-epoch-{}".format(now, epoch))

                # Make plots of the histograms of the learned sweep parameters.
                fig_hist, axes_hist = plot_utils.inferer_plot_quant_hist(sess, X_test, X,
                                                                         output, hyperparams)
                # wandb.log({"hist_plot": fig_hist}, step=epoch)
                fig_hist.savefig("plots/fig-{}/hist-epoch-{}".format(now, epoch))

        print("[" + "=" * 25 + "]", end="\t")

        # ---------------------- Log results ---------------------- #
        # calculate loss
        loss_train = loss_total.eval(feed_dict={X: X_train_aug, y: y_train}) / X_train_aug.shape[0]
        loss_test = loss_total.eval(feed_dict={X: X_test, y: y_test}) / X_test.shape[0]
        print("Epoch {:5}\tWall: {} \tTraining: {:.4e}\tTesting: {:.4e}"
              .format(epoch, datetime.utcnow().strftime("%H:%M:%S"),
                      loss_train, loss_test))
        wandb.log({'loss_train': loss_train, 'loss_test': loss_test}, step=epoch)
        saver.save(sess, "./saved_models/model-{}-final.ckpt".format(now))

        # Calculate RMS percent error of quantities. Quants output order is: ne, Vp, Te.
        # We can divide by y_test_scaled because all these quantities must be greater than 0.
        quants = output.eval(feed_dict={X: X_test}) * y_ptp + y_mean
        y_test_scaled = y_test * y_ptp + y_mean
        per_ne = ((quants[:, 0] - y_test_scaled[:, 0]) / y_test_scaled[:, 0] * 100 - 100)
        per_Vp = ((quants[:, 1] - y_test_scaled[:, 1]) / y_test_scaled[:, 1] * 100 - 100)
        per_Te = ((quants[:, 2] - y_test_scaled[:, 2]) / y_test_scaled[:, 2] * 100 - 100)
        print("RMS: \tne: {:03.1f}%\tVp: {:03.1f}%\tTe: {:03.1f}%\t"
              .format(np.std(per_ne), np.std(per_Vp), np.std(per_Te)))
        wandb.log({'RMS_pct_ne': np.std(per_ne), 'RMS_pct_Vp': np.std(per_Vp),
                   'RMS_pct_Te': np.std(per_Te)}, step=epoch)

        # Make plots comparing learned parameters to the actual ones.
        fig_compare, axes = plot_utils. \
            inferer_plot_comparison_including_vsweep(sess, X, X_test, X_mean, X_ptp, output,
                                                     y_mean, y_ptp, hyperparams)
        wandb.log({"comaprison_plot": fig_compare}, step=epoch)
        fig_compare.savefig("plots/fig-{}/compare".format(now))

        # Show the worst performing fits (may not implement this).
        # fig_worst, axes = plot_utils.plot_worst(sess, X_train, X, output, hyperparams)
        # fig_worst.savefig("plots/fig-{}/worst".format(now))

        # Make plots of the histograms of the learned sweep parameters.
        # The conversion that WandB does to plotly really sucks.
        fig_hist, axes_hist = plot_utils.inferer_plot_quant_hist(sess, X_test, X,
                                                                 output, hyperparams)
        wandb.log({"hist_plot": fig_hist}, step=epoch)
        fig_hist.savefig("plots/fig-{}/hist".format(now))

        # Log tensorflow graph and variables.
        final_checkpoint_name = "./saved_models/model-{}-final.ckpt".format(now)
        wandb.save(final_checkpoint_name + ".index")
        wandb.save(final_checkpoint_name + ".meta")
        wandb.save(final_checkpoint_name + ".data-00000-of-00001")
        best_checkpoint_name = "./saved_models/model-{}-best.ckpt".format(now)
        wandb.save(best_checkpoint_name + ".index")
        wandb.save(best_checkpoint_name + ".meta")
        wandb.save(best_checkpoint_name + ".data-00000-of-00001")


if __name__ == '__main__':
    hyperparams = {'n_inputs': 500,
                   'scale': 0.1,
                   'learning_rate': 5e-7,
                   'momentum': 0.99,
                   'frac_train': 0.6,
                   'frac_test': 0.2,
                   'frac_valid': 0.2,  # This is actually unused.
                   'batch_size': 512,
                   'steps': 2000,
                   'seed': 42}
    wandb.init(project="sweep-langmuir-ml", sync_tensorboard=True, config=hyperparams)
    train(hyperparams, debug=True)
