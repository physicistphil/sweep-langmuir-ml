# This file should be run from the folder its in. E.g.: python train.py

import tensorflow as tf
import numpy as np
from datetime import datetime

# Modify log levels to keep console clean.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# Custom tools from other directories
import sys
sys.path.append('../utilities')
import preprocess
import generate

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
    num_batches = int(np.ceil(data_train.shape[0] / hyperparams['batch_size']))
    num_test_batches = int(np.ceil(data_test.shape[0] / hyperparams['batch_size']))
    # Maybe delete after? Oh well.

    # Build the model to train.
    model = build_full.Model()
    # Build the data pipeline
    model.build_data_pipeline(hyperparams, data_mean, data_ptp)

    # Build the component that compresses the sweep into some latent space
    model.build_CNN(hyperparams, model.data_X_train, model.data_X_test)

    # Build the network that translates from the CNN output to the Langmuir sweep model
    model.build_linear_translator(hyperparams, model.CNN_output)

    # Get the surrogate model and connect it to our current model.
    surrogate_X = tf.concat([model.phys_input,
                             model.X[:, 0:hyperparams['n_inputs']] *
                             model.data_ptp[0:hyperparams['n_inputs']] +
                             model.data_mean[0:hyperparams['n_inputs']]], 1)
    surrogate_path = "./saved_models/" + hyperparams['surrogate_model'] + ".ckpt"
    surr_import = tf.train.import_meta_graph("./saved_models/" + hyperparams['surrogate_model'] +
                                             ".ckpt.meta",
                                             input_map={"data/X": surrogate_X},
                                             import_scope="surrogate")
    surr_output = tf.get_default_graph().get_tensor_by_name("surrogate/nn/output:0")
    scalefactor = tf.get_default_graph().get_tensor_by_name("surrogate/const/scalefactor:0")

    # So we can get the physical plasma parameters out from the model.
    model.build_plasma_info(scalefactor)
    # Process the curve coming out of the sweep model.
    model.build_theory_processor(hyperparams, surr_output, stop_gradient=False)
    # The discrepancy model tries to fit the difference between the original and analytic curves.
    # model.build_discrepancy_model(hyperparams, model.X[:, 0:hyperparams['n_inputs']],
    #                               (model.X[:, hyperparams['n_inputs']:] -
    #                                model.processed_theory))
    # Instead learn the discrepancy from the CNN output (not on the difference).
    model.build_learned_discrepancy_model(hyperparams, model.CNN_output)

    # Remove surrogate model from the list of trainable variables (to pass in to the optimizer)
    training_vars = tf.trainable_variables()
    removelist = tf.trainable_variables(scope='surrogate')
    for var in removelist:
        training_vars.remove(var)
    model.vars = training_vars
    # Calculate all the losses (full curve, theory fit, discrepancy size, regularization)
    model.build_loss(hyperparams, model.X[:, hyperparams['n_inputs']:],
                     model.processed_theory, model.discrepancy_output)

    # Log values of gradients and variables for tensorboard.
    for grad, var in model.grads:
        if grad is not None and var is not None:
            tf.compat.v1.summary.histogram("gradients/" + var.name, grad)
            tf.compat.v1.summary.histogram("variables/" + var.name, var)
    for var in tf.trainable_variables():
        tf.compat.v1.summary.histogram("trainables/" + var.name, var)

    # Initialize configuration and variables.
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    # for batch normalization updates
    extra_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        # ---------------------- Initialize everything ---------------------- #
        # Initialize variables
        init.run()
        # Initialize data iterators
        sess.run(model.data_train_iter.initializer, feed_dict={model.data_train: data_train})
        sess.run(model.data_test_iter.initializer, feed_dict={model.data_test: data_test})

        # Make sure to restore full model before the surrogate so that the variables for the
        #   surrogate model are restored correctly. This order also allows different surrogate
        #   models to be used
        if hyperparams['restore']:
            model.load_model(sess, "./saved_models/" + hyperparams['restore_model'] + ".ckpt")

        # Restore surrogate model parameters
        surr_import.restore(sess, surrogate_path)

        # Use regex to remove the surrogate model ops from the summary results so that
        #   the data pipeline ops of the surrogate model are not ran.
        summaries_op = tf.compat.v1.summary.merge_all(scope="^((?!surrogate).)*$")
        summary_writer = tf.compat.v1.summary.FileWriter("summaries/sum-" + now, graph=sess.graph)
        best_loss = np.finfo(np.float32).max

        # ---------------------- Begin training ---------------------- #
        for epoch in range(hyperparams['steps']):
            # Initialize the training data iterator.
            temp_loss_train = 0
            for i in range(num_batches):
                _, _, loss_train = sess.run([model.training_op, extra_update_ops, model.loss_total],
                                            feed_dict={model.training: True})
                # Keep track of average loss
                temp_loss_train += loss_train / num_batches
            loss_train = temp_loss_train

            print("[" + "=" * int(20.0 * (epoch % 10) / 10.0) +
                  " " * (20 - int(20.0 * (epoch % 10) / 10.0)) +
                  "]", end="\t")
            print(("Epoch {:5}\tT: {} \tLoss train: {:.3e}")
                  .format(epoch, datetime.utcnow().strftime("%H:%M:%S"), loss_train), end="")
            print("\r", end="")

            # Every 10th epoch (or last epoch), calculate testing loss and maybe save the model.
            if epoch % 10 == 0 or epoch == hyperparams['steps'] - 1:
                # Write summaries
                try:
                    summary = sess.run(summaries_op, feed_dict={model.training: False})
                    summary_writer.add_summary(summary, epoch)
                except tf.errors.InvalidArgumentError:
                    print("NaN in summary histogram; no summary generated.")

                # Evaluate on the test set
                loss_test = 0
                for i in range(num_test_batches):
                    temp_loss_test = (sess.run(model.loss_total, feed_dict={model.training: False}))
                    loss_test += temp_loss_test / num_test_batches

                print("[" + "=" * 20 + "]", end="\t")
                print(("Epoch {:5}\tT: {} \tLoss train: {:.3e} \tLoss test: {:.3e}")
                      .format(epoch, datetime.utcnow().strftime("%H:%M:%S"), loss_train, loss_test))
                print("[" + " " * 5 + "Saving...." + " " * 5 + "]", end="\r")

                wandb.log({'loss_train': loss_train}, step=epoch)
                wandb.log({'loss_test': loss_test}, step=epoch)
                model.plot_comparison(sess, data_test, hyperparams, fig_path, epoch)

                if loss_test < best_loss:
                    best_loss = loss_test
                    saver.save(sess, "./saved_models/model-{}-best.ckpt".format(now))

                print("[" + " " * 20 + "]", end="\r")

            if epoch % 100 == 0:
                saver.save(sess, "./saved_models/model-{}-epoch-{}.ckpt".format(now, epoch))
                wandb.log({"Comparison plot":
                           wandb.Image(fig_path + 'full-compare-epoch-{}.png'.format(epoch))})

        # ---------------------- Log results, make figures ---------------------- #
        wandb.log({"Comparison plot":
                   wandb.Image(fig_path + 'full-compare-epoch-{}.png'.format(epoch))})
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
                   'filters': 4,
                   'size_diff': 64,
                   'n_output': 256,
                   # Loss scaling weights (please normalize)
                   'loss_rebuilt': 2.0,  # Controls the influence of the rebuilt curve
                   'loss_theory': 0.2,  # Controls how tightly the theory must fit the original
                   'loss_discrepancy': 0.6,  # Controls how small the discrepancy must be
                   'l2_CNN': 0.00,
                   'l2_discrepancy': 4.0,
                   'l2_translator': 0.00,
                   # Optimization hyperparamters
                   'learning_rate': 1e-6,
                   'momentum': 0.99,
                   'batch_momentum': 0.99,
                   'batch_size': 1024,
                   # Data paramters
                   # 'num_batches': 16,  # Number of batches trained in each epoch.
                   'frac_train': 0.8,
                   'frac_test': 0.2,
                   # Training info
                   'steps': 100,
                   'seed': 137,
                   'restore': True,
                   'restore_model': "model-20200407190928-final",
                   'surrogate_model': "model-20200327211709-final"
                   }
    wandb.init(project="sweep-langmuir-ml", sync_tensorboard=True, config=hyperparams,
               notes="Restored from last (20200407190928)")
    train(hyperparams)
