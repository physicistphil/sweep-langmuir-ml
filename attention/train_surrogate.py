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

# This constant scales the input of the surrogate model (we store it in the graph to use later).
#   The factor makes training easier (because gradients and values remain sane) and allows us
#   to use mixed precision without NaNs.
scalefactor = np.array([1e-17, 1e-1, 1e19, 1e0])


# Note: the generator will not be serialized with the graph when saved (see TF docs).
def trace_generator(hyperparams, limit=-1):
    ne_range = np.array([1e15, 1e18])
    Vp_range = np.array([-50, 40])
    e = 1.602e-19  # Elementary charge
    Te_range = np.array([0.1, 12]) * e
    vsweep_lower_range = np.array([-120, Vp_range[0] - 10])
    vsweep_upper_range = np.array([Vp_range[1], 100])
    # One call of the generator is one batch.
    size = hyperparams['batch_size']
    n_inputs = hyperparams['n_inputs']
    S = 2e-6

    me = 9.109e-31

    print("ne_range: ", ne_range,
          "\nVp_range: ", Vp_range,
          "\nTe_range: ", Te_range,
          "\nvsweep_lower_range: ", vsweep_lower_range,
          "\nvsweep_upper_range: ", vsweep_upper_range)

    # Iteration counter
    i = 0
    while i != limit:
        # print("Generating {}".format(i))
        # Change the seed so that new curves are created with each call of the generator, but
        #   determinism is preserved.
        # We must not modify hyperparams or else we'll end up with a seed that exceeds 2^32 - 1.
        #   See https://oeis.org/A000217
        # new_hyperparams = {'seed': hyperparams['seed'] + i, 'n_inputs': hyperparams['n_inputs']}
        i += 1
        # ne, Vp, Te, vsweep, current \
        #     = generate.generate_random_traces_from_array(ne_range, Vp_range, Te_range,
        #                                                  vsweep_lower_range, vsweep_upper_range,
        #                                                  new_hyperparams, size, S=S)

        np.random.seed(i + 0)
        vsweep_lower = np.random.uniform(vsweep_lower_range[0], vsweep_lower_range[1], size)
        np.random.seed(i + 1)
        vsweep_upper = np.random.uniform(vsweep_upper_range[0], vsweep_upper_range[1], size)
        vsweep = np.ndarray(shape=(size, n_inputs))
        vsweep = np.linspace(vsweep_lower, vsweep_upper, 256, axis=1)

        np.random.seed(i + 2)
        ne = (np.exp(np.random.uniform(np.log(ne_range[0]), np.log(ne_range[1]), (size, 1))))
        np.random.seed(i + 3)
        Vp = (np.random.uniform(Vp_range[0], Vp_range[1], (size, 1)))
        np.random.seed(i + 4)
        Te = (np.random.uniform(Te_range[0], Te_range[1], (size, 1)))

        I_esat = S * ne * e / np.sqrt(2 * np.pi * me)
        current = I_esat * np.sqrt(Te) * np.exp(-e * (Vp - vsweep) / Te)
        esat_condition = Vp < vsweep
        current[esat_condition] = np.repeat(I_esat * np.sqrt(Te), n_inputs, axis=1)[esat_condition]

        yield np.hstack((ne * scalefactor[0],
                         Vp * scalefactor[1],
                         Te * scalefactor[2],
                         vsweep * scalefactor[3])), current


def train(hyperparams):
    tf.compat.v1.disable_v2_behavior()

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    wandb.log({"now": now}, step=0)
    os.mkdir("plots/fig-{}".format(now))
    fig_path = "plots/fig-{}/".format(now)

    # Build the model to train.
    model = build_surrogate.Model()
    model.build_data_pipeline(hyperparams, trace_generator, scalefactor)
    model.build_dense_NN(hyperparams, model.data_X, model.data_y)

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
        # Initialize the data iterator. We're not initalizing in each epoch because we want to
        #   keep generating new data from the generator.
        sess.run(model.data_iter.initializer)

        if hyperparams['restore']:
            model.load_dense_model(sess, "./saved_models/" + hyperparams['restore_model'] + ".ckpt")

        summaries_op = tf.compat.v1.summary.merge_all()
        summary_writer = tf.compat.v1.summary.FileWriter("summaries/sum-" + now, graph=sess.graph)
        best_loss = np.finfo(np.float32).max

        # ---------------------- Begin training ---------------------- #
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
                  "]", end="\t")
            print(("Epoch {:5}\tT: {} \tp_tr: {:.3e}")
                  .format(epoch, datetime.utcnow().strftime("%H:%M:%S"), loss_train), end="")
            print("\r", end="")

            # At multiples of 10, we take a break and save our model.
            if epoch % 100 == 0:
                print("[" + "=" * 20 + "]", end="\t")
                print(("Epoch {:5}\tT: {} \tp_tr: {:.3e}")
                      .format(epoch, datetime.utcnow().strftime("%H:%M:%S"), loss_train))
                print("[" + " " * 5 + "Saving...." + " " * 5 + "]", end="\r")

                wandb.log({'loss_train': loss_train}, step=epoch)
                model.plot_comparison(sess, hyperparams, fig_path, epoch)
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
        model.plot_comparison(sess, hyperparams, fig_path, epoch)
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
                   'size_l1': 40,
                   'size_l2': 40,
                   # 'size_lh': 20,
                   'n_output': 256,
                   # Optimization hyperparamters
                   'learning_rate': 2e-6,
                   'momentum': 0.99,
                   'batch_momentum': 0.99,
                   'l2_scale': 0.00,
                   'batch_size': 1024,  # Actual batch size is n_inputs * batch_size (see build_NN)
                   # Data paramters
                   'num_batches': 8,  # Number of batches trained in each epoch.
                   # Training info
                   'steps': 25000,
                   'seed': 0,
                   'restore': True,
                   'restore_model': "model-20201023183911-final"
                   }
    wandb.init(project="sweep-langmuir-ml", sync_tensorboard=True, config=hyperparams,
               notes="Cont on 20201023183911. Expanded parameter sweep range compared to 20200327211709. Generate sweeps with no flat bits.")

    print("Hyperparameters:")
    for param in hyperparams.items():
        print("{}: {}".format(param[0], param[1]))
    print("\n")

    train(hyperparams)
