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
sys.path.append('.../utilities')
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


def train(hyperparams, debug=False):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    wandb.log({"now": now}, step=0)
    os.mkdir("plots/fig-{}".format(now))
    fig_path = "plots/fig-{}/".format(now)

    # Gather all the data
    data_train, data_test, data_valid, data_mean, data_ptp = get_real_data(hyperparams)

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
        ae_best_loss = 1000

        for epoch in range(hyperparams['steps']):

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

            print("[" + "=" * int(20.0 * (epoch % 10) / 10.0) +
                  " " * (20 - int(20.0 * (epoch % 10) / 10.0)) +
                  "]", end="")
            print("\r", end="")

            # loss_test = (loss_total.eval(feed_dict={X: X_test, y: y_test}) /
            #              X_test.shape[0])
            # print("loss: {}, epoch: {}".format(loss_test, epoch))

            if epoch % 10 == 0:
                print("[" + "=" * 20 + "]", end="\t")

                ae_loss_train = (ae_loss_total.eval(feed_dict={X: data_train[0:batch_size],
                                                               y: np.zeros((batch_size, 3))}) /
                                 batch_size)
                ae_loss_test = (ae_loss_total.eval(feed_dict={X: data_test[0:batch_size],
                                                              y: np.zeros((batch_size, 3))}) /
                                batch_size)
                wandb.log({'ae_loss_train': ae_loss_train,
                           'ae_loss_test': ae_loss_test}, step=epoch)

                print(("Epoch {:5}\tT: {} \ta_tr: {:.3e}\ta_te: {:.3e}")
                      .format(epoch, datetime.utcnow().strftime("%H:%M:%S"),
                              ae_loss_train, ae_loss_test))

                if ae_loss_test < ae_best_loss:
                    ae_best_loss = ae_loss_test
                    saver.save(sess, "./saved_models/model-{}-best.ckpt".format(now))

            if epoch % 100 == 0:
                saver.save(sess, "./saved_models/model-{}-epoch-{}.ckpt".format(now, epoch))

            # Make plots comparing learned parameters to the actual ones.
            if epoch % 100 == 0:  # Changed this to 100 from 1000 because we have much more data.

                fig_compare_ae, axes_ae = plot_utils. \
                    autoencoder_plot_comparison(sess, data_test[0:batch_size], X, ae_output,
                                                hyperparams)
                fig_compare_ae.savefig(fig_path + "ae_compare-epoch-{}".format(epoch))

                # Close all the figures so that memory can be freed.
                plt.close('all')

        print("[" + "=" * 20 + "]", end="\t")

        # ---------------------- Log results ---------------------- #
        # calculate loss
        ae_loss_train = (ae_loss_total.eval(feed_dict={X: data_train[0:batch_size],
                                                       y: np.zeros((batch_size, 3))}) /
                         batch_size)
        ae_loss_test = (ae_loss_total.eval(feed_dict={X: data_test[0:batch_size],
                                                      y: np.zeros((batch_size, 3))}) /
                        batch_size)
        wandb.log({'ae_loss_train': ae_loss_train,
                   'ae_loss_test': ae_loss_test}, step=epoch)

        print(("Epoch {:5}\tT: {} \ta_tr: {:.3e}\ta_te: {:.3e}")
              .format(epoch, datetime.utcnow().strftime("%H:%M:%S"),
                      ae_loss_train, ae_loss_test))

        saver.save(sess, "./saved_models/model-{}-final.ckpt".format(now))

        # ---------------------- Make figures ---------------------- #
        fig_compare_ae, axes_ae = plot_utils. \
            autoencoder_plot_comparison(sess, data_test[0:batch_size], X, ae_output, hyperparams, y)
        fig_compare_ae.savefig(fig_path + "ae_compare".format(now))
        wandb.log({"ae_comaprison_plot": [wandb.Image(fig_compare_ae)]}, step=epoch)

        # Show the worst performing fits (may not implement this).
        # fig_worst, axes = plot_utils.plot_worst(sess, X_train, X, output, hyperparams)
        # fig_worst.savefig("plots/fig-{}/worst".format(now))

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
                   'n_phys_inputs': 3,
                   'filters': 3,
                   'size_li': 50,
                   'switch_num': 1,  # Number of epochs to train ae or inferer before switching
                   'freeze_ae': True,
                   # Optimization hyperparamters
                   'learning_rate': 1e-6,
                   'momentum': 0.99,
                   'l2_scale': 0.1,
                   'batch_size': 1024,
                   # Data paramters
                   'num_examples': 2 ** 14,  # There are 16320 real traces.
                   'frac_train': 0.6,
                   'frac_test': 0.2,
                   'frac_valid': 0.2,  # This is actually unused lol.
                   # Augmentation parameters (for synthetic traces)
                   'offset_scale': 0.0,
                   'noise_scale': 0.4,
                   # Training info
                   'steps': 20000,
                   'seed': 42,
                   }
    wandb.init(project="sweep-langmuir-ml", sync_tensorboard=True, config=hyperparams,)
    train(hyperparams, debug=True)