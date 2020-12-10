import tensorflow as tf
import numpy as np
from datetime import datetime

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Modify log levels to keep console clean if you want.
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'  # If you want to use tensor cores.

# From the current directory
import build
# import build_analytic  # for the analytical model, which I haven't used in forever.
import get_data

# weights and biases -- ML experiment tracker.
import wandb


def train(hyperparams):
    # I have not yet updated the code to be native to tensorflow 2.
    tf.compat.v1.disable_v2_behavior()

    # All files associated with an ML training run are identified by the time they were launched.
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    wandb.log({"now": now}, step=0)
    os.mkdir("plots/fig-{}".format(now))
    fig_path = "plots/fig-{}/".format(now)

    # Gather all the data, both synthetic and real, combined and shuffled.
    data_train, data_test, data_valid, data_mean, data_ptp = get_data.sample_datasets(hyperparams)
    num_batches = int(np.ceil(data_train.shape[0] / hyperparams['batch_size']))
    num_test_batches = int(np.ceil(data_test.shape[0] / hyperparams['batch_size']))
    # Log the number of examples used to wandb
    wandb.log({"num_ex_actual": data_train.shape[0] + data_test.shape[0] + data_valid.shape[0]},
              step=0)

    # Instantiate the ML model class.
    model = build.Model()
    # Build the data pipeline for faster loading / prefetching.
    model.build_data_pipeline(hyperparams, data_mean, data_ptp)
    # Switch between train and test set that depends on a "training" parameter that's fed in later.
    model.build_X_switch(hyperparams)
    # Learn / extract the features from the Langmuir sweeps.
    model.build_feature_extractor(hyperparams)
    # Learn what parts of the sweep to focus on.
    model.build_attention_focuser(hyperparams)
    # Build the network that translates from the C/NN output to the Langmuir sweep model.
    # This component outputs scaled ne, Vp, and Te numbers (but not in physical units).
    model.build_physics_translator(hyperparams)

    # Get the vsweeps from our data and convert them from normalized to real physics units (Volts).
    vsweep = (model.X[:, 0:hyperparams['n_inputs']] * model.data_ptp[0:hyperparams['n_inputs']] +
              model.data_mean[0:hyperparams['n_inputs']])
    # The surrogate model provides the theoretical mapping from vsweep and plasma parameters
    #   to current.
    # Get the surrogate model and connect it to our current model.
    # Only provide ne, Vp, and Te, and the voltage sweep to the surrogate model.
    surrogate_X = tf.concat([model.phys_input[:, 0:3], vsweep], 1)
    surrogate_path = "./saved_models/" + hyperparams['surrogate_model'] + ".ckpt"
    # The surrogate model's input data it's tensor 'X' -- we need to map that to our current model
    #   which is surrogate_X using the input_map keyword. Use surrogate scope for easy var handling.
    surr_import = tf.compat.v1.train.import_meta_graph("./saved_models/" +
                                                       hyperparams['surrogate_model'] +
                                                       ".ckpt.meta",
                                                       input_map={"data/X": surrogate_X},
                                                       import_scope="surrogate")
    # Get the surrogate output tensor so we can connect it to our current model.
    surr_output = tf.compat.v1.get_default_graph().get_tensor_by_name("surrogate/nn/output:0")
    # Scalefactor from the surrogate model is for ne, Vp, Te, and vsweep (in that order).
    # We need this scalefactor so that we can properly translate from phys_input (not in
    #   physical units) to units understandable to humans.
    scalefactor = tf.compat.v1.get_default_graph().get_tensor_by_name("surrogate/const/scalefactor:0")
    # Use the scalefactor that the surrogate model uses to get the physical plasma parameters
    #   out from our model.
    model.build_plasma_info(scalefactor)
    # Process the curve coming out of the sweep model. It gets normalized by our data ptp and mean
    #   so that it can be directly compared with our data at a reasonable scale.
    model.build_surrogate_output_normalizer(hyperparams, surr_output)
    # Remove surrogate model from the list of trainable variables (to pass in to the optimizer) so
    #   that we don't train our surrogate model (which is trained separately).
    training_vars = tf.compat.v1.trainable_variables()
    removelist = tf.compat.v1.trainable_variables(scope='surrogate')
    for var in removelist:
        training_vars.remove(var)
    model.vars = training_vars

    # Build the classification portion of the model.
    model.build_classifier(hyperparams)

    # Calculate the loss for our model
    model.build_loss(hyperparams, scalefactor)

    # Log values of gradients and variables for tensorboard. If grad and var are None it'll crash.
    # "Summaries" give us the distribution of weights, variables, and other info we want.
    for grad, var in model.grads:
        if grad is not None and var is not None:
            tf.compat.v1.summary.histogram("gradients/" + var.name, grad)
            tf.compat.v1.summary.histogram("variables/" + var.name, var)
    # Add all trainable variables to our histogram so we have a list that doesn't include the
    #   surrogate model parameters.
    for var in tf.compat.v1.trainable_variables():
        tf.compat.v1.summary.histogram("trainables/" + var.name, var)

    # Initialize tensorflow configuration and variables.
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    # For batch normalization updates, if used.
    extra_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    # Allow GPU memory to grow instead of allocating all memory on the GPU from the start.
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        # ---------------------- Initialize everything ---------------------- #
        # Initialize variables.
        init.run()
        # Initialize data iterators.
        sess.run(model.data_train_iter.initializer, feed_dict={model.data_train: data_train})
        sess.run(model.data_test_iter.initializer, feed_dict={model.data_test: data_test})

        # Make sure to restore full model before the surrogate so that the variables for the
        #   surrogate model are restored correctly. This order also allows different surrogate
        #   models to be used.
        if hyperparams['restore']:
            model.load_model(sess, "./saved_models/" + hyperparams['restore_model'] + ".ckpt")
        # Restore surrogate model parameters
        surr_import.restore(sess, surrogate_path)
        # Use regex to remove the surrogate model ops from the summary results so that
        #   the data pipeline ops of the surrogate model are not run.
        summaries_op = tf.compat.v1.summary.merge_all(scope="^((?!surrogate).)*$")
        summary_writer = tf.compat.v1.summary.FileWriter("summaries/sum-" + now, graph=sess.graph)
        # Set our "best loss" number to the maximum floating point number.
        best_loss = np.finfo(np.float32).max

        # ---------------------- Training loop ---------------------- #
        for epoch in range(hyperparams['steps']):
            temp_loss_train = 0  # Accumulator for training loss.
            # Iterate through batches.
            for i in range(num_batches):
                _, _, loss_train = sess.run([model.training_op, extra_update_ops, model.loss_total],
                                            feed_dict={model.training: True})
                temp_loss_train += loss_train / num_batches  # Keep track of average loss.
            loss_train = temp_loss_train
            # Stupid workaround for wandb complaining about writing to older history rows.
            if epoch % 10 != 0 and epoch != hyperparams['steps'] - 1:
                wandb.log({'loss_train': loss_train}, step=epoch)
            # Fancy-pants terminal output.
            print("[" + "=" * int(20.0 * (epoch % 10) / 10.0) +
                  " " * (20 - int(20.0 * (epoch % 10) / 10.0)) +
                  "]", end="\t")
            print(("Epoch {:5}\tT: {} \tLoss train: {:.3e}")
                  .format(epoch, datetime.utcnow().strftime("%H:%M:%S"), loss_train), end="")
            print("\r", end="")

            # Every 10th epoch (or last epoch), calculate testing loss and save the model.
            if True: #epoch % 10 == 0 or epoch == hyperparams['steps'] - 1:
                # Write summaries.
                try:
                    summary = sess.run(summaries_op, feed_dict={model.training: False})
                    summary_writer.add_summary(summary, epoch)
                except tf.errors.InvalidArgumentError:
                    print("NaN in summary histogram; no summary generated.")
                # Evaluate model on the test set.
                loss_test = 0
                for i in range(num_test_batches):
                    temp_loss_test = (sess.run(model.loss_total, feed_dict={model.training: False}))
                    loss_test += temp_loss_test / num_test_batches
                # Fancy-pants terminal output.
                print("[" + "=" * 20 + "]", end="\t")
                print(("Epoch {:5}\tT: {} \tLoss train: {:.3e} \tLoss test: {:.3e}")
                      .format(epoch, datetime.utcnow().strftime("%H:%M:%S"), loss_train, loss_test))
                print("[" + " " * 5 + "Saving...." + " " * 5 + "]", end="\r")
                # Log loss results to weights and biases.
                wandb.log({'loss_train': loss_train}, step=epoch)
                wandb.log({'loss_test': loss_test}, step=epoch)
                # Use our testing data to plot model performance. This is the primary way I judge
                #   model performance. This takes one batch of the test set, which means that
                #   for each epoch this displays a different batch of the test set because one
                #   full cycle + 1 of the test set was iterated through.
                model.plot_comparison(sess, hyperparams, fig_path, epoch)
                # Save the best model so far as determined by testing loss.
                if loss_test < best_loss:
                    best_loss = loss_test
                    saver.save(sess, "./saved_models/model-{}-best.ckpt".format(now))
                # Fancy-pants terminal output.
                print("[" + " " * 20 + "]", end="\r")
            # Log the comparison plot to wandb every 10th epoch.
            if epoch % 10 == 0:
                saver.save(sess, "./saved_models/model-{}-epoch-{}.ckpt".format(now, epoch))
                wandb.log({"Comparison plot":
                           wandb.Image(fig_path + 'full-compare-epoch-{}.png'.format(epoch))})

        # ---------------------- Log results ---------------------- #
        wandb.log({"Comparison plot":
                   wandb.Image(fig_path + 'full-compare-epoch-{}.png'.format(epoch))})
        saver.save(sess, "./saved_models/model-{}-final.ckpt".format(now))  # Save final model.

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
    hyperparams = {'n_inputs': 256,  # Number of points in the voltage and current sweeps.
                   # One flag is to enable / disable physical parameter loss.
                   #   This is used for the synthetic dataset, telling the model that it has
                   #   physical parameters (value = 1) that it can use in the loss function.
                   #   Real datasets have a value of 0.
                   # The other flag is used for labeling bad sweeps. 1=bad, 0=good.
                   'n_flag_inputs': 2,
                   'n_phys_inputs': 3,  # n_e, V_p and T_e (for now).
                   # CNN parameters.
                   'attn_filters': 16,  # Number of filters used in the attention portion.
                   'feat_filters': 8,  # Number of filters used in the feature extractor.
                   'filters': 8,  # Number of filters used in the translator portion.
                   'class_width': 16,
                   # Loss weights.
                   'loss_rebuilt': 6.0,  # Controls the influence of the rebuilt curve error.
                   'loss_physics': 6.0,  # Strength of phys param matching for synthetic sweeps.
                   'loss_misclass': 0.0,
                   'loss_bad_penalty': 0.0,
                   'l1_CNN_output': 0.0,  # L1 on output of CNN (phys_input).
                   'l2_CNN': 1e-4,  # L2 regularization on CNNs in the model.
                   'l2_discrepancy': 1.0,
                   'l2_translator': 0.00,  # L2 regularization on the FFNN portion of translator.
                   'l2_classifier': 0.00,
                   # Optimization hyperparamters for the adam optimizer
                   'learning_rate': 1e-5,
                   'beta1': 0.9,
                   'beta2': 0.999,
                   'epsilon': 1e-8,
                   # Batch info.
                   'batch_momentum': 0.95,  # Momentum used in batch normalization calculation.
                   'batch_size': 128,  # 128 seems optimal.
                   # Training info.
                   'steps': 30,
                   'seed': 137,
                   'restore': False,  # Controls if we restore on an existing model.
                   'restore_model': "model-AAA-final",  # Which model to restore on.
                   'surrogate_model': "model-20201026164454-final",  # Which surrogate model to use.
                   # Data parameters.
                   'frac_train': 0.8,  # Fraction of data to train on.
                   'frac_test': 0.2,  # Fraction of data to test on.
                   'datasets': ['mirror1',  # Real / experiment datasets to train on.
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
                   # Which synthetic dataset(s) to use.
                   'datasets_synthetic': ['15-18_-50-40_0-1-12_-120-100_corrupt-esat_0-5-5_normed_w-rootfunc-v2'],
                   # Which bad datasets to use (for the classification portion).
                   'datasets_bad': ['data_synthetic/bad_sweeps_01'],
                   # Examples to use from _each_ dataset (use all in dataset if # too large).
                   'num_examples': 1 * 2 ** 15,  # 15, 17, 15
                   # Examples to use from _each_ dataset (use all in dataset if # too large)
                   'num_synthetic_examples': 1 * int(1.5 * 2 ** 17),
                   # Examples to use from the bad examples dataset
                   'num_bad_examples': 2 ** 17,
                   # The scale of the random offset applied to the synthetic data.
                   'offset_scale': 0.0,
                   # Scale of noise applied to synthetic data. 1.0 =  1x the height of an exmaple.
                   'noise_scale': 0.07
                   }
    # Notes for the training run that show up on wandb.
    notes = "First tests with badness classifier (measuring things...)"
    wandb.init(project="sweep-langmuir-ml", sync_tensorboard=True, config=hyperparams,
               notes=notes)
    # Print hyperparameters and notes so they show up in the terminal.
    print("Hyperparameters:")
    for param in hyperparams.items():
        print("{}: {}".format(param[0], param[1]))
    print("\n")
    print(notes)
    print("\n")
    # Begin training.
    train(hyperparams)
