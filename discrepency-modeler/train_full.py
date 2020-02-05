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
sys.path.append('../utilities')
import preprocess
import generate
import plot_utils

# From the directory
import build_full
import build_analytic  # for the analytical model

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


def train(hyperparams):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    wandb.log({"now": now}, step=0)
    os.mkdir("plots/fig-{}".format(now))
    fig_path = "plots/fig-{}/".format(now)

    # Gather all the data. Stack the mean and ptp as if they're examples, but trim them off when
    #   we actually build the model.
    data_train, data_test, data_valid, data_mean, data_ptp = get_real_data(hyperparams)
    data_input = np.vstack([data_train, data_mean, data_ptp])
    print(data_input.shape)
    # Maybe delete after? Oh well.

    # Build the model to train.
    model = build_full.Model()
    # Build the data pipeline
    model.build_data_pipeline(hyperparams)
    # Build the component that compresses the sweep into some latent space
    model.build_CNN(hyperparams, model.data_X)
    # Build the network that translates from the CNN output to the Langmuir sweep model
    model.build_linear_translator(hyperparams, model.CNN_output)
    # Build the analytical Langmuir sweep model.
    analytic_model = build_analytic.Model()
    analytic_model.build_analytical_model(hyperparams, model.layer_convert_activation,
                                          model.data_X[:, 0:hyperparams['n_inputs']] *
                                          model.data_ptp[0:hyperparams['n_inputs']] +
                                          model.data_mean[0:hyperparams['n_inputs']])
    # Process the curve coming out of the sweep model.
    model.build_theory_processor(hyperparams, analytic_model.phys_output, stop_gradient=False)
    # The discrepancy model tries to fit the difference between the original and analytic curves.
    model.build_discrepancy_model(hyperparams, model.data_X[:, 0:hyperparams['n_inputs']],
                                  (model.data_X[:, hyperparams['n_inputs']:] -
                                   model.processed_theory))
    # Calculate all the losses (full curve, theory fit, discrepancy size, regularization)
    model.build_loss(hyperparams, model.data_X[:, hyperparams['n_inputs']:],
                     model.processed_theory, model.discrepancy_output)

    # Log values of gradients and variables for tensorboard.
    for grad, var in model.grads:
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

    with tf.compat.v1.Session(config=config) as sess:

        # ---------------------- Initialize everything ---------------------- #
        init.run()
        # Initialize the data iterator.
        sess.run(model.data_iter.initializer, feed_dict={model.data_input: data_input})

        if hyperparams['restore']:
            model.load_model(sess, "./saved_models/" + hyperparams['restore_model'] + ".ckpt")

        summaries_op = tf.compat.v1.summary.merge_all()
        summary_writer = tf.compat.v1.summary.FileWriter("summaries/sum-" + now, graph=sess.graph)
        best_loss = np.finfo(np.float32).max

        # ---------------------- Begin training ---------------------- #
        num_batches = int(np.ceil(data_input.shape[0] / hyperparams['batch_size']))
        for epoch in range(hyperparams['steps']):
            for i in range(num_batches):
                _, _, loss_train = sess.run([model.training_op, extra_update_ops, model.loss_total],
                                            feed_dict={model.training: True,
                                                       model.data_input: data_input})

                if i == 0 and epoch % 10 == 0:
                    try:
                        summary = sess.run(summaries_op, feed_dict={model.training: True,
                                                                    model.data_input: data_input})
                        summary_writer.add_summary(summary, epoch)
                    except tf.errors.InvalidArgumentError:
                        print("NaN in summary histogram; no summary generated.")

            print("[" + "=" * int(20.0 * (epoch % 10) / 10.0) +
                  " " * (20 - int(20.0 * (epoch % 10) / 10.0)) +
                  "]", end="\t")
            print(("Epoch {:5}\tT: {} \tp_tr: {:.3e}")
                  .format(epoch, datetime.utcnow().strftime("%H:%M:%S"), loss_train), end="")
            print("\r", end="")

            # At multiples of 10, we take a break and save our model.
            if epoch % 10 == 0:
                print("[" + "=" * 20 + "]", end="\t")
                print(("Epoch {:5}\tT: {} \tp_tr: {:.3e}")
                      .format(epoch, datetime.utcnow().strftime("%H:%M:%S"), loss_train))
                print("[" + " " * 5 + "Saving...." + " " * 5 + "]", end="\r")

                wandb.log({'loss_train': loss_train}, step=epoch)
                model.plot_comparison(sess, data_input, hyperparams, fig_path, epoch)
                if best_loss < best_loss:
                    best_loss = best_loss
                    saver.save(sess, "./saved_models/model-{}-best.ckpt".format(now))

                print("[" + " " * 20 + "]", end="\r")

            if epoch % 100 == 0:
                saver.save(sess, "./saved_models/model-{}-epoch-{}.ckpt".format(now, epoch))

        print("[" + "=" * 20 + "]", end="\t")
        print(("Epoch {:5}\tT: {} \tp_tr: {:.3e}")
              .format(epoch, datetime.utcnow().strftime("%H:%M:%S"), loss_train))

        # ---------------------- Log results, make figures ---------------------- #
        wandb.log({'loss_train': loss_train}, step=epoch)
        model.plot_comparison(sess, data_input, hyperparams, fig_path, epoch)
        saver.save(sess, "./saved_models/model-{}-final.ckpt".format(now))

        # Log tensorflow checkpoints (takes up a lot of space).
        final_checkpoint_name = "./saved_models/model-{}-final.ckpt".format(now)
        wandb.save(final_checkpoint_name + ".index")
        wandb.save(final_checkpoint_name + ".meta")
        wandb.save(final_checkpoint_name + ".data-00000-of-00001")
        best_checkpoint_name = "./saved_models/model-{}-best.ckpt".format(now)
        wandb.save(best_checkpoint_name + ".index")
        wandb.save(best_checkpoint_name + ".meta")
        wandb.save(best_checkpoint_name + ".data-00000-of-00001")


if __name__ == '__main__':
    hyperparams = {'n_inputs': 256,  # Number of points to define the voltage sweep.
                   'n_phys_inputs': 3,  # n_e, V_p and T_e (for now).
                   # 'size_l1': 50,
                   # 'size_l2': 50,
                   # 'size_trans': 50,
                   'filters': 3,
                   'size_diff': 20,
                   'n_output': 256,
                   # Optimization hyperparamters
                   'learning_rate': 1e-7,
                   'momentum': 0.99,
                   'batch_momentum': 0.99,
                   'l2_scale': 0.00,
                   'batch_size': 1024,  # Actual batch size is n_inputs * batch_size (see build_NN)
                   # Data paramters
                   # 'num_batches': 16,  # Number of batches trained in each epoch.
                   'frac_train': 0.6,
                   'frac_test': 0.2,
                   # Training info
                   'steps': 1000,
                   'seed': 42,
                   'restore': False,
                   'restore_model': "model-WWWWWWWWWWWWWWW-final"
                   }
    wandb.init(project="sweep-langmuir-ml", sync_tensorboard=True, config=hyperparams,)
    train(hyperparams)
