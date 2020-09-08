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
import get_data

# weights and biases -- ML experiment tracker
import wandb


def train(hyperparams):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    wandb.log({"now": now}, step=0)
    os.mkdir("plots/fig-{}".format(now))
    fig_path = "plots/fig-{}/".format(now)

    # Gather all the data.
    data_train, data_test, data_valid, data_mean, data_ptp = get_data.sample_datasets(hyperparams)
    num_batches = int(np.ceil(data_train.shape[0] / hyperparams['batch_size']))
    num_test_batches = int(np.ceil(data_test.shape[0] / hyperparams['batch_size']))
    # Maybe delete after? Oh well.
    wandb.log({"num_ex_actual": data_train.shape[0] + data_test.shape[0] + data_valid.shape[0]},
              step=0)

    # Build the model to train.
    model = build_full.Model()
    # Build the data pipeline
    model.build_data_pipeline(hyperparams, data_mean, data_ptp)

    # Build the component that compresses the sweep into some latent space
    # Keep in mind that the values after n_inputs * 2 are the physical parameters.
    model.build_CNN(hyperparams, model.data_X_train, model.data_X_test)

    # Build the network that translates from the CNN output to the Langmuir sweep model
    model.build_linear_translator(hyperparams, model.CNN_output)

    vsweep = (model.X[:, 0:hyperparams['n_inputs']] * model.data_ptp[0:hyperparams['n_inputs']] +
              model.data_mean[0:hyperparams['n_inputs']])

    # Get the surrogate model and connect it to our current model (and providing vsweep).
    surrogate_X = tf.concat([model.phys_input[:, 0:3],  # only provide ne, Vp, and Te
                             vsweep], 1)
    surrogate_path = "./saved_models/" + hyperparams['surrogate_model'] + ".ckpt"
    surr_import = tf.train.import_meta_graph("./saved_models/" + hyperparams['surrogate_model'] +
                                             ".ckpt.meta",
                                             input_map={"data/X": surrogate_X},
                                             import_scope="surrogate")
    surr_output = tf.get_default_graph().get_tensor_by_name("surrogate/nn/output:0")
    # Scalefactor from the surrogate model is ne, Vp, Te, and vsweep
    surr_scalefactor = tf.get_default_graph().get_tensor_by_name("surrogate/const/scalefactor:0")

    # Build the monoenergetic primary electron model
    # monoenergetic_scalefactor = tf.constant([1e-14, 1 / 5.0])
    # scalefactor = tf.concat([surr_scalefactor, monoenergetic_scalefactor], 0)
    scalefactor = surr_scalefactor
    # model.build_monoenergetic_electron_model(hyperparams, model.phys_input,
    #                                          vsweep, scalefactor)

    # So we can get the physical plasma parameters out from the model.
    # model.build_plasma_info(scalefactor)
    model.build_plasma_info(scalefactor)
    # Process the curve coming out of the sweep model.
    model.build_theory_processor(hyperparams, surr_output,  # + model.monoenergetic_output,
                                 stop_gradient=False)
    # Instead learn the discrepancy from the CNN output (not on the difference).
    # model.build_learned_discrepancy_model(hyperparams, model.CNN_output)

    # Remove surrogate model from the list of trainable variables (to pass in to the optimizer)
    training_vars = tf.trainable_variables()
    removelist = tf.trainable_variables(scope='surrogate')
    for var in removelist:
        training_vars.remove(var)
    model.vars = training_vars
    # Calculate all the losses (full curve, theory fit, discrepancy size, regularization, physics)
    model.build_loss(hyperparams, model.X[:, hyperparams['n_inputs']:],
                     model.processed_theory,
                     0.0,  # model.discrepancy_output,
                     model.X_phys, scalefactor)

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
            # Stupid workaround for wandb complaining about writing to older history rows
            if epoch % 10 != 0 and epoch != hyperparams['steps'] - 1:
                wandb.log({'loss_train': loss_train}, step=epoch)
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
                model.plot_comparison(sess, hyperparams, fig_path, epoch)

                if loss_test < best_loss:
                    best_loss = loss_test
                    saver.save(sess, "./saved_models/model-{}-best.ckpt".format(now))

                print("[" + " " * 20 + "]", end="\r")

            if epoch % 10 == 0:
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
                   'n_flag_inputs': 1,  # Flag to enable / disable physical parameter loss.
                   'n_phys_inputs': 3,  # n_e, V_p and T_e, n_p, and E_p
                   # 'size_l1': 50,
                   # 'size_l2': 50,
                   # 'size_trans': 50,
                   'filters': 8,
                   'size_diff': 64,
                   'n_output': 256,
                   # Loss scaling weights (rebuilt, theory, and discrepancy are normalized)
                   'loss_rebuilt': 0.2,  # 2 Controls the influence of the rebuilt curve
                   # 'loss_theory': 0.00,  # 0.01 Controls how tightly the theory must fit the original
                   # 'loss_discrepancy': 0.0,  # 0.001 Controls how small the discrepancy must be
                   'loss_physics': 2.0,  # Not included in norm. Loss weight of phys params.
                   'loss_phys_penalty': 0.0,  # Penalize size of physical params
                   'l1_CNN_output': 0.0,  # l1 on output of CNN
                   'l2_CNN': 0.001,
                   'l2_discrepancy': 1.0,
                   'l2_translator': 0.00,
                   'loss_scale': 10.0,  # Controls the scale of the sqrt loss function
                   # Optimization hyperparamters
                   'learning_rate': 3e-4,
                   # 'momentum': 0.99,
                   'beta1': 0.9,
                   'beta2': 0.999,
                   'epsilon': 1e-8,
                   'batch_momentum': 0.9,
                   'batch_size': 64,
                   # Training info
                   'steps': 60,
                   'seed': 137,
                   'restore': True,
                   'restore_model': "model-20200907230327-final",
                   'surrogate_model': "model-20200327211709-final",
                   # Data parameters
                   'frac_train': 0.8,
                   'frac_test': 0.2,
                   'datasets': ['mirror1',
                                'mirror2',
                                'mirror3',
                                'mirror4',
                                # 'mirror5',  # set aside for validation
                                'edge1',
                                'edge2',
                                'core',
                                'walt1',
                                'mirror1_avg',
                                'mirror2_avg',
                                'mirror3_avg',
                                'mirror4_avg',
                                # 'mirror5_avg',  # set aside for validation
                                'edge1_avg',
                                'edge2_avg',
                                'core_avg',
                                'walt1_avg'],
                   'datasets_synthetic': [#'16-18_0-20_0-5-10_-50--20_20-60_corrupt-esat',
                                          #'15-18_-30-30_0-1-10_-100--20_20-100_corrupt-esat',
                                          #'15-18_-30-20_0-1-10_-100-100_corrupt-esat-continuous',
                                          '15-18_-50-40_0-1-12_-120-100_corrupt-esat_0-5-2'],
                   'num_examples': 1 * 2 ** 16,  # Examples from each dataset (use all if # too large)
                   'num_synthetic_examples': int(1.0 * 2 ** 20),  # See comment above
                   'offset_scale': 0.03,
                   'noise_scale': 0.03
                   }

    wandb.init(project="sweep-langmuir-ml", sync_tensorboard=True, config=hyperparams,
               notes="Long run -- contiue off of 20200907230327, + real data, 0.05 noise and offset")

    print("Hyperparameters:")
    for param in hyperparams.items():
        print("{}: {}".format(param[0], param[1]))
    print("\n")

    train(hyperparams)
