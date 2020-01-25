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

# From the inferer directory
import build_surrogate

# weights and biases -- ML experiment tracker
import wandb


# Note: the generator will not be serialized with the graph when saved (see TF docs).
def trace_generator(hyperparams, limit=-1):
    ne_range = np.array([1e16, 1e18])
    Vp_range = np.array([0, 20])
    e = 1.602e-19  # Elementary charge
    Te_range = np.array([0.5, 10]) * e
    vsweep_lower_range = np.array([-50, -20])
    vsweep_upper_range = np.array([50, 100])
    # One call of the generator is one batch.
    size = hyperparams['batch_size']
    S = 2e-6

    # Iteration counter
    i = 0
    while i != limit:
        # print("Generating {}".format(i))
        # Change the seed so that new curves are created with each call of the generator, but
        #   determinism is preserved.
        hyperparams['seed'] = hyperparams['seed'] + i
        i += 1
        ne, Vp, Te, vsweep, current \
            = generate.generate_random_traces_from_array(ne_range, Vp_range, Te_range,
                                                         vsweep_lower_range, vsweep_upper_range,
                                                         hyperparams, size, S=S)
        yield np.hstack((ne[:, np.newaxis], Vp[:, np.newaxis], Te[:, np.newaxis], vsweep)), current


def train(hyperparams):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    wandb.log({"now": now}, step=0)
    os.mkdir("plots/fig-{}".format(now))
    fig_path = "plots/fig-{}/".format(now)

    # Build the model to train.
    model = build_surrogate.DenseNN()
    model.build_data_pipeline(hyperparams, trace_generator)
    model.build_NN(hyperparams, model.data_X, model.data_y)

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

    # ---------------------- Begin training ---------------------- #
    with tf.compat.v1.Session(config=config) as sess:
        init.run()
        # Initialize the data iterator. We're not initalizing in each epoch because we want to
        #   keep generating new data from the generator.
        sess.run(model.data_iter.initializer)

        summaries_op = tf.compat.v1.summary.merge_all()
        summary_writer = tf.compat.v1.summary.FileWriter("summaries/sum-" + now, graph=sess.graph)
        best_loss = 1000

        for epoch in range(hyperparams['steps']):

            # Train the physics portion of the network.
            # We have no testing loss because our data is randomly generated (no overfitting)
            for i in range((hyperparams['num_batches'])):
                _, _, loss_train = sess.run([model.training_op, extra_update_ops, model.loss_total],
                                            feed_dict={model.training: True})

                if i == 0 and epoch % 10 == 0:
                    try:
                        summary = sess.run(summaries_op, feed_dict={model.training: True})
                        summary_writer.add_summary(summary, epoch)
                    except tf.errors.InvalidArgumentError:
                        print("NaN in summary histogram; no summary generated.")

            print("[" + "=" * int(20.0 * (epoch % 10) / 10.0) +
                  " " * (20 - int(20.0 * (epoch % 10) / 10.0)) +
                  "]", end="")
            print("\r", end="")

            if epoch % 10 == 0:
                print("[" + "=" * 20 + "]", end="\t")

                wandb.log({'loss_train': loss_train}, step=epoch)

                print(("Epoch {:5}\tT: {} \tp_tr: {:.3e}")
                      .format(epoch, datetime.utcnow().strftime("%H:%M:%S"), loss_train))

                if best_loss < best_loss:
                    best_loss = best_loss
                    saver.save(sess, "./saved_models/model-{}-best.ckpt".format(now))

                model.plot_comparison(sess, hyperparams, fig_path, epoch)

            if epoch % 100 == 0:
                saver.save(sess, "./saved_models/model-{}-epoch-{}.ckpt".format(now, epoch))

            # Make plots comparing learned parameters to the actual ones.
            if epoch % 100 == 0:  # Changed this to 100 from 1000 because we have much more data.

                # fig_compare_ae, axes_ae = plot_utils. \
                #     phys_plot_comparison(sess, data_test[0:batch_size], data_mean, data_ptp,
                #                          X, X_mean, X_ptp, phys_output, phys_input,
                #                          hyperparams)
                # fig_compare_ae.savefig(fig_path + "phys_compare-epoch-{}".format(epoch))

                # Close all the figures so that memory can be freed.
                plt.close('all')

        print("[" + "=" * 20 + "]", end="\t")

        # ---------------------- Log results ---------------------- #
        wandb.log({'loss_train': loss_train}, step=epoch)

        print(("Epoch {:5}\tT: {} \tp_tr: {:.3e}")
              .format(epoch, datetime.utcnow().strftime("%H:%M:%S"), loss_train))

        saver.save(sess, "./saved_models/model-{}-final.ckpt".format(now))

        # ---------------------- Make figures ---------------------- #
        # fig_compare_phys, axes_phys = plot_utils. \
        #     phys_plot_comparison(sess, data_test[0:batch_size], data_mean, data_ptp,
        #                          X, X_mean, X_ptp, phys_output, phys_input, hyperparams)
        # fig_compare_phys.savefig(fig_path + "phys_compare".format(now))
        # wandb.log({"phys_comaprison_plot": [wandb.Image(fig_compare_phys)]}, step=epoch)

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
    hyperparams = {'n_inputs': 256,  # Number of points to define the voltage sweep.
                   'n_phys_inputs': 3,  # n_e, V_p and T_e (for now).
                   'size_l1': 50,
                   'size_l2': 50,
                   # 'size_lh': 20,
                   'n_output': 256,
                   # Optimization hyperparamters
                   'learning_rate': 1e-7,
                   'momentum': 0.99,
                   'batch_momentum': 0.99,
                   'l2_scale': 0.0,
                   'batch_size': 256,  # Actual batch size is n_inputs * batch_size (see build_NN)
                   # Data paramters
                   'num_batches': 8,  # Number of batches trained in each epoch.
                   # Training info
                   'steps': 500,
                   'seed': 42,
                   }
    wandb.init(project="sweep-langmuir-ml", sync_tensorboard=True, config=hyperparams,)
    train(hyperparams)
