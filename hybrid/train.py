# This file should be run from the folder its in. E.g.: python train.py

import tensorflow as tf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Modify log levels to keep console clean.
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Custom tools from other directories
import sys
sys.path.append('/home/phil/Desktop/sweeps/sweep-langmuir-ml/utilities')
import preprocess
import generate
import plot_utils

# From the inferer directory
import build_graph

# weights and biases -- ML experiment tracker
import wandb


# Get data from real experiments (so far just from the mirror dataset).
def get_real_data(hyperparams):
    print("Getting data...", end=" ")
    sys.stdout.flush()
    n_inputs = hyperparams['n_inputs']
    signal = preprocess.get_mirror_data_with_sweeps(n_inputs)

    # Find the voltage sweep and current means and peak-to-peaks so the model is easier to train.
    vsweep_mean = np.full(hyperparams['n_inputs'], np.mean(signal[:, 0:n_inputs]))
    vsweep_ptp = np.full(hyperparams['n_inputs'], np.ptp(signal[:, 0:n_inputs]))
    current_mean = np.full(hyperparams['n_inputs'], np.mean(signal[:, n_inputs:]))
    current_ptp = np.full(hyperparams['n_inputs'], np.ptp(signal[:, n_inputs:]))
    # Combine the two so we have a nice neat X, y, and scalings tuple returned by the function.
    data_mean = np.concatenate((vsweep_mean, current_mean))
    data_ptp = np.concatenate((vsweep_ptp, current_ptp))

    # Voltage and current sweeps are already concatenated.
    # Centering and scaling the input so that it's easier to train.
    data = (signal - data_mean) / data_ptp
    data_train, data_test, data_valid = preprocess.shuffle_split_data(data, hyperparams)
    print("Done.")

    return data_train, data_test, data_valid, data_mean, data_ptp


def gather_random_synthetic_scaled_data(hyperparams):
    print("Generating traces...", end=" ", flush=True)

    # Number of examples to generate. There are 3264 * 5 real ones from the mirror dataset.
    size = hyperparams['num_examples']

    ne_range = np.array([1e15, 1e18])
    Vp_range = np.array([1, 20])
    e = 1.602e-19  # Elementary charge
    Te_range = np.array([0.5, 5]) * e  # We're defining it in terms of eV because it's comfortable.
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
    # We're also centering and scaling the input so that it's easier to train.
    X = (np.concatenate((vsweep, current), axis=1) - X_mean) / X_ptp
    X_train, X_test, X_valid = preprocess.shuffle_split_data(X, hyperparams)

    # Center and rescale the sweep parameters, otherwise the model is impossible to train.
    y = np.stack((ne, Vp, Te)).transpose()
    y_mean = np.mean(y, axis=0)
    y_ptp = np.ptp(y, axis=0)
    y = (y - y_mean) / y_ptp
    y_train, y_test, y_valid = preprocess.shuffle_split_data(y, hyperparams)
    print("Done.")

    return X_train, X_test, X_valid, X_mean, X_ptp, y_train, y_test, y_valid, y_mean, y_ptp


# Mix the generated synthetic sweeps and real data and return the combined set.
# We have 16320 real traces, so we probably want a similar amount of synthetic ones if we want to
#   train them on equal footing.
def gather_mixed_data(hyperparams):
    print("Getting mixed data...", end=" ")
    sys.stdout.flush()
    n_inputs = hyperparams['n_inputs']
    signal = preprocess.get_mirror_data_with_sweeps(n_inputs)

    # Find the voltage sweep and current means and peak-to-peaks so the model is easier to train.
    vsweep_mean = np.full(hyperparams['n_inputs'], np.mean(signal[:, 0:n_inputs]))
    vsweep_ptp = np.full(hyperparams['n_inputs'], np.ptp(signal[:, 0:n_inputs]))
    current_mean = np.full(hyperparams['n_inputs'], np.mean(signal[:, n_inputs:]))
    current_ptp = np.full(hyperparams['n_inputs'], np.ptp(signal[:, n_inputs:]))
    # Combine the two so we have a nice neat X, y, and scalings tuple returned by the function.
    data_mean = np.concatenate((vsweep_mean, current_mean))
    data_ptp = np.concatenate((vsweep_ptp, current_ptp))

    # Get synthetic traces.
    size = hyperparams['num_examples']
    ne_range = np.array([1e16, 1e18])
    Vp_range = np.array([0, 20])
    e = 1.602e-19  # Elementary charge
    Te_range = np.array([0.5, 10]) * e  # We're defining it in terms of eV because it's comfortable.
    S = 2e-6  # Probe area in m^2
    # Voltages used when collecting real sweeps are within this range.
    vsweep_lower_range = np.array([-50, -20])
    vsweep_upper_range = np.array([50, 100])
    ne, Vp, Te, vsweep, current \
        = generate.generate_random_traces_from_array(ne_range, Vp_range, Te_range,
                                                     vsweep_lower_range, vsweep_upper_range,
                                                     hyperparams, size, S=S)
    # Concatenate synthetic sweep and traces.
    X = np.concatenate((vsweep, current), axis=1)
    # Merge real and synthetic datasets
    data = np.concatenate((signal, X), axis=0)

    # Voltage and current sweeps are already concatenated.
    # Centering and scaling the input so that it's easier to train.
    data = (data - data_mean) / data_ptp
    data_train, data_test, data_valid = preprocess.shuffle_split_data(data, hyperparams)
    print("Done.")

    return data_train, data_test, data_valid, data_mean, data_ptp


def train(hyperparams, debug=False):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    wandb.log({"now": now}, step=0)
    os.mkdir("plots/fig-{}".format(now))
    fig_path = "plots/fig-{}/".format(now)

    # Gather all the data
    # data_train, data_test, data_valid, data_mean, data_ptp = get_real_data(hyperparams)
    data_train, data_test, data_valid, data_mean, data_ptp = gather_mixed_data(hyperparams)
    X_train, X_test, X_valid, X_mean, X_ptp, y_train, y_test, y_valid, y_mean, y_ptp \
        = gather_random_synthetic_scaled_data(hyperparams)
    # data_train = data_test = data_valid = data_mean = data_ptp = \
        # X_train = X_test = X_valid = X_mean = X_ptp \
        # = np.ones((1024, 1000))
    # y_train = y_test = y_valid = y_mean = y_ptp = np.ones((1024, 3))


    # Build the models to train on.
    if not debug:
        (ae_training_op, infer_training_op, X, y, training, ae_output, infer_output,
         ae_loss_total, infer_loss_total) = build_graph.make_small_nn(hyperparams, size_output=3)
    else:
        (ae_training_op, infer_training_op, X, y, training, ae_output, infer_output,
         ae_loss_total, infer_loss_total,
         ae_grads, infer_grads) = build_graph.make_conv_nn(hyperparams, size_output=3, debug=debug)

        for grad, var in ae_grads:
            if grad is not None and var is not None:
                tf.compat.v1.summary.histogram("gradients/" + var.name, grad)
                tf.compat.v1.summary.histogram("variables/" + var.name, var)
        for grad, var in infer_grads:
            if grad is not None and var is not None:
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
        batch_size = hyperparams['batch_size']

        init.run()
        summaries_op = tf.compat.v1.summary.merge_all()
        summary_writer = tf.compat.v1.summary.FileWriter("summaries/sum-" + now, graph=sess.graph)
        infer_best_loss = 1000

        # Augment our test data.
        X_test = preprocess.add_offset(X_test, hyperparams, epoch=0)
        # X_test = preprocess.add_noise(X_test, hyperparams, epoch=0)
        X_test = preprocess.add_real_noise(X_test, hyperparams, epoch=0)

        # Apply random offset to learn invariance.
        X_train_aug = preprocess.add_offset(X_train, hyperparams, epoch=0)
        # Apply noise.
        # X_train_aug = preprocess.add_noise(X_train_aug, hyperparams, epoch=epoch)
        X_train_aug = preprocess.add_real_noise(X_train_aug, hyperparams, epoch=0)
        for epoch in range(hyperparams['steps']):

            # Alternate training between autoencoder and inferer.
            if epoch % (hyperparams['switch_num'] * 2) < hyperparams['switch_num']:
                # Train the autoencoder.
                for i in range(data_train.shape[0] // batch_size):
                    data_batch = data_train[i * batch_size:(i + 1) * batch_size]
                    sess.run([ae_training_op, extra_update_ops],
                             feed_dict={X: data_batch, y: np.zeros((batch_size, 3)),
                                        training: True})
                    if i == 0 and epoch % 10 == 0 and debug:
                        summary = sess.run(summaries_op,
                                           feed_dict={X: data_train[0:batch_size],
                                                      y: np.zeros((batch_size, 3)),
                                                      training: True})
                        summary_writer.add_summary(summary, epoch)
                if (data_train.shape[0] % batch_size) != 0:
                    data_batch = data_train[(i + 1) * batch_size:]
                    sess.run([ae_training_op, extra_update_ops],
                             feed_dict={X: data_batch, y: np.zeros((batch_size, 3)),
                                        training: True})

            else:
                # Train the inferer.
                for i in range(X_train_aug.shape[0] // batch_size):
                    X_batch = X_train_aug[i * batch_size:
                                          (i + 1) * batch_size]
                    y_batch = y_train[i * batch_size:
                                      (i + 1) * batch_size]
                    sess.run([infer_training_op, extra_update_ops],
                             feed_dict={X: X_batch, y: y_batch, training: True})
                    if i == 0 and epoch % 10 == 0 and debug:
                        summary = sess.run(summaries_op,
                                           feed_dict={X: X_train_aug[0:batch_size],
                                                      y: y_train[0:batch_size], training: True})
                        summary_writer.add_summary(summary, epoch)
                if (X_train_aug.shape[0] % batch_size) != 0:
                    X_batch = X_train_aug[(i + 1) * batch_size:]
                    y_batch = y_train[(i + 1) * batch_size:]
                    sess.run([infer_training_op, extra_update_ops],
                             feed_dict={X: X_batch, y: y_batch, training: True})

            print("[" + "=" * int(20.0 * (epoch % 10) / 10.0) +
                  " " * (20 - int(20.0 * (epoch % 10) / 10.0)) +
                  "]", end="")
            print("\r", end="")

            # loss_test = (loss_total.eval(feed_dict={X: X_test, y: y_test}) /
            #              X_test.shape[0])
            # print("loss: {}, epoch: {}".format(loss_test, epoch))

            if epoch % 10 == 0:
                print("[" + "=" * 20 + "]", end="\t")

                infer_loss_train = (infer_loss_total.eval(feed_dict={X: X_train_aug[0:batch_size],
                                                                     y: y_train[0:batch_size]}) /
                                    batch_size)
                infer_loss_test = (infer_loss_total.eval(feed_dict={X: X_test[0:batch_size],
                                                                    y: y_test[0:batch_size]}) /
                                   batch_size)
                wandb.log({'infer_loss_train': infer_loss_train,
                           'infer_loss_test': infer_loss_test}, step=epoch)
                ae_loss_train = (ae_loss_total.eval(feed_dict={X: data_train[0:batch_size],
                                                               y: np.zeros((batch_size, 3))}) /
                                 batch_size)
                ae_loss_test = (ae_loss_total.eval(feed_dict={X: data_test[0:batch_size],
                                                              y: np.zeros((batch_size, 3))}) /
                                batch_size)
                wandb.log({'ae_loss_train': ae_loss_train,
                           'ae_loss_test': ae_loss_test}, step=epoch)

                print(("Epoch {:5}\tT: {} \ti_tr: {:.3e}\ti_te: {:.3e} " +
                       "\ta_tr: {:.3e}\ta_te: {:.3e}")
                      .format(epoch, datetime.utcnow().strftime("%H:%M:%S"),
                              infer_loss_train, infer_loss_test, ae_loss_train, ae_loss_test))

                if infer_loss_test < infer_best_loss:
                    infer_best_loss = infer_loss_test
                    saver.save(sess, "./saved_models/model-{}-best.ckpt".format(now))

            if epoch % 100 == 0:
                saver.save(sess, "./saved_models/model-{}.ckpt".format(now))

            # Make plots comparing learned parameters to the actual ones.
            if epoch % 100 == 0:  # Changed this to 100 from 1000 because we have much more data.
                fig_compare, axes = plot_utils. \
                    inferer_plot_comparison_including_vsweep(sess, X, X_test[0:batch_size],
                                                             X_mean, X_ptp, infer_output,
                                                             y_mean, y_ptp, hyperparams)
                # wandb.log({"comaprison_plot": fig_compare}, step=epoch)
                fig_compare.savefig(fig_path + "infer_compare-epoch-{}".format(epoch))

                fig_compare_ae, axes_ae = plot_utils. \
                    autoencoder_plot_comparison(sess, data_test[0:batch_size], X, ae_output,
                                                hyperparams, y)
                fig_compare_ae.savefig(fig_path + "ae_compare-epoch-{}".format(epoch))

                # Make plots of the histograms of the learned sweep parameters.
                fig_hist, axes_hist = plot_utils.inferer_plot_quant_hist(sess, X_test[0:batch_size],
                                                                         X, infer_output,
                                                                         hyperparams)
                # wandb.log({"hist_plot": fig_hist}, step=epoch)
                fig_hist.savefig(fig_path + "hist-epoch-{}".format(epoch))
                # Close all the figures so that memory can be freed.
                plt.close('all')

                # Calculate RMS percent error of quantities. Quants output order is: ne, Vp, Te.
                # We can divide by y_test_scaled because it's always > 0.
                quants = infer_output.eval(feed_dict={X: X_test[0:batch_size]}) * y_ptp + y_mean
                y_test_scaled = y_test[0:batch_size] * y_ptp + y_mean
                per_ne = ((quants[:, 0] - y_test_scaled[:, 0]) / y_test_scaled[:, 0] * 100 - 100)
                per_Vp = ((quants[:, 1] - y_test_scaled[:, 1]) / y_test_scaled[:, 1] * 100 - 100)
                per_Te = ((quants[:, 2] - y_test_scaled[:, 2]) / y_test_scaled[:, 2] * 100 - 100)
                print("RMS: \tne: {:03.1f}%\tVp: {:03.1f}%\tTe: {:03.1f}%\t"
                      .format(np.std(per_ne), np.std(per_Vp), np.std(per_Te)))
                wandb.log({'RMS_pct_ne': np.std(per_ne), 'RMS_pct_Vp': np.std(per_Vp),
                           'RMS_pct_Te': np.std(per_Te)}, step=epoch)

        print("[" + "=" * 20 + "]", end="\t")

        # ---------------------- Log results ---------------------- #
        # calculate loss
        infer_loss_train = (infer_loss_total.eval(feed_dict={X: X_train_aug[0:batch_size],
                                                             y: y_train[0:batch_size]}) /
                            batch_size)
        infer_loss_test = (infer_loss_total.eval(feed_dict={X: X_test[0:batch_size],
                                                            y: y_test[0:batch_size]}) /
                           batch_size)
        wandb.log({'infer_loss_train': infer_loss_train,
                   'infer_loss_test': infer_loss_test}, step=epoch)
        ae_loss_train = (ae_loss_total.eval(feed_dict={X: data_train[0:batch_size],
                                                       y: np.zeros((batch_size, 3))}) /
                         batch_size)
        ae_loss_test = (ae_loss_total.eval(feed_dict={X: data_test[0:batch_size],
                                                      y: np.zeros((batch_size, 3))}) /
                        batch_size)
        wandb.log({'ae_loss_train': ae_loss_train,
                   'ae_loss_test': ae_loss_test}, step=epoch)

        print(("Epoch {:5}\tT: {} \ti_tr: {:.3e}\ti_te: {:.3e} " +
               "\ta_tr: {:.3e}\ta_te: {:.3e}")
              .format(epoch, datetime.utcnow().strftime("%H:%M:%S"),
                      infer_loss_train, infer_loss_test, ae_loss_train, ae_loss_test))

        saver.save(sess, "./saved_models/model-{}-final.ckpt".format(now))

        # Calculate RMS percent error of quantities. Quants output order is: ne, Vp, Te.
        # We can divide by y_test_scaled because all these quantities must be greater than 0.
        quants = infer_output.eval(feed_dict={X: X_test[0:batch_size]}) * y_ptp + y_mean
        y_test_scaled = y_test[0:batch_size] * y_ptp + y_mean
        per_ne = ((quants[:, 0] - y_test_scaled[:, 0]) / y_test_scaled[:, 0] * 100 - 100)
        per_Vp = ((quants[:, 1] - y_test_scaled[:, 1]) / y_test_scaled[:, 1] * 100 - 100)
        per_Te = ((quants[:, 2] - y_test_scaled[:, 2]) / y_test_scaled[:, 2] * 100 - 100)
        print("RMS: \tne: {:03.1f}%\tVp: {:03.1f}%\tTe: {:03.1f}%\t"
              .format(np.std(per_ne), np.std(per_Vp), np.std(per_Te)))
        wandb.log({'RMS_pct_ne': np.std(per_ne), 'RMS_pct_Vp': np.std(per_Vp),
                   'RMS_pct_Te': np.std(per_Te)}, step=epoch)

        # ---------------------- Make figures ---------------------- #
        # Make plots comparing learned parameters to the actual ones.
        fig_compare, axes = plot_utils. \
            inferer_plot_comparison_including_vsweep(sess, X, X_test[0:batch_size], X_mean,
                                                     X_ptp, infer_output,
                                                     y_mean, y_ptp, hyperparams)
        fig_compare.savefig(fig_path + "infer_compare".format(now))
        wandb.log({"infer_comaprison_plot": [wandb.Image(fig_compare)]}, step=epoch)

        fig_compare_ae, axes_ae = plot_utils. \
            autoencoder_plot_comparison(sess, data_test[0:batch_size], X, ae_output, hyperparams, y)
        fig_compare_ae.savefig(fig_path + "ae_compare".format(now))
        wandb.log({"ae_comaprison_plot": [wandb.Image(fig_compare_ae)]}, step=epoch)

        # Show the worst performing fits (may not implement this).
        # fig_worst, axes = plot_utils.plot_worst(sess, X_train, X, output, hyperparams)
        # fig_worst.savefig("plots/fig-{}/worst".format(now))

        # Make plots of the histograms of the learned sweep parameters.
        # The conversion that WandB does to plotly really sucks, so I'll use PIL for PNGs
        fig_hist, axes_hist = plot_utils.inferer_plot_quant_hist(sess, X_test[0:batch_size], X,
                                                                 infer_output, hyperparams)
        fig_hist.savefig(fig_path + "hist".format(now))
        wandb.log({"hist_plot": [wandb.Image(fig_hist)]}, step=epoch)

        # Log tensorflow checkpoints (takes up a lot of space).
        # final_checkpoint_name = "./saved_models/model-{}-final.ckpt".format(now)
        # wandb.save(final_checkpoint_name + ".index")
        # wandb.save(final_checkpoint_name + ".meta")
        # wandb.save(final_checkpoint_name + ".data-00000-of-00001")
        # best_checkpoint_name = "./saved_models/model-{}-best.ckpt".format(now)
        # wandb.save(best_checkpoint_name + ".index")
        # wandb.save(best_checkpoint_name + ".meta")
        # wandb.save(best_checkpoint_name + ".data-00000-of-00001")


if __name__ == '__main__':
    hyperparams = {'n_inputs': 500,
                   # 'size_l1': 50,
                   # 'size_l2': 50,
                   # 'size_lh': 20,
                   # 'size_li': 10,
                   'switch_num': 1,  # Number of epochs to train ae or inferer before switching
                   'freeze_ae': True,
                   # Optimization hyperparamters
                   'learning_rate': 1e-6,
                   'momentum': 0.99,
                   'l2_scale': 0.1,
                   'batch_size': 1024,
                   # Data paramters
                   'num_examples': 2 ** 16,  # There are 16320 real traces.
                   'frac_train': 0.6,
                   'frac_test': 0.2,
                   'frac_valid': 0.2,  # This is actually unused lol.
                   # Augmentation parameters (for synthetic traces)
                   'offset_scale': 0.0,
                   'noise_scale': 0.4,
                   # Training info
                   'steps': 2000,
                   'seed': 42,
                   }
    wandb.init(project="sweep-langmuir-ml", sync_tensorboard=True, config=hyperparams,)
    train(hyperparams, debug=True)
