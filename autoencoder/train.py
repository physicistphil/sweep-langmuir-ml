from comet_ml import Experiment

import tensorflow as tf
import numpy as np
from datetime import datetime

# custom tools
import sys
sys.path.append('/home/phil/Desktop/sweeps/sweep-langmuir-ml/data_preprocessing')
import preprocess
import plot_utils
import build_graph


def train(experiment, hyperparams):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    training_op, loss_total, X, training, output = build_graph.deep_7(hyperparams)

    np.random.seed(hyperparams['seed'])
    data = preprocess.get_mirror_data(hyperparams['n_inputs'])
    np.random.shuffle(data)
    data_size = data.shape[0]
    data_train = data[0:int(data_size * hyperparams['frac_train']), :]
    data_test = data[int(data_size * hyperparams['frac_train']):
                     int(data_size * (hyperparams['frac_test'] + hyperparams['frac_train'])), :]
    data_valid = data[int(data_size * (hyperparams['frac_test'] + hyperparams['frac_train'])):, :]

    experiment.log_dataset_hash(data)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # for batch normalization updates
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init.run()
        experiment.set_model_graph(sess.graph)

        for epoch in range(hyperparams['steps']):
            for i in range(data_train.shape[0] // hyperparams['batch_size']):
                X_batch = data_train[i * hyperparams['batch_size']:
                                     (i + 1) * hyperparams['batch_size']]
                sess.run([training_op, extra_update_ops], feed_dict={X: X_batch, training: True})
            X_batch = data_train[(i + 1) * hyperparams['batch_size']:]
            sess.run([training_op, extra_update_ops], feed_dict={X: X_batch, training: True})

            if epoch % 10 == 0:
                loss_train = loss_total.eval(feed_dict={X: data_train}) / data_train.shape[0]
                loss_test = loss_total.eval(feed_dict={X: data_test}) / data_test.shape[0]
                with experiment.train():
                    experiment.log_metric('Loss', loss_train, step=epoch)
                with experiment.test():
                    experiment.log_metric('Loss', loss_test, step=epoch)
                print("Epoch {:5}\tWall: {} \tTraining loss: {:.4e}\tTesting loss: {:.4e}"
                      .format(epoch, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                              loss_train, loss_test))

            if epoch % 100 == 0:
                saver.save(sess, "./saved_models/ae-001-{}.ckpt".format(now))

        experiment.set_step(epoch)

        loss_train = loss_total.eval(feed_dict={X: data_train}) / data_train.shape[0]
        loss_test = loss_total.eval(feed_dict={X: data_test}) / data_test.shape[0]
        print("Epoch {:5}\tWall: {} \tTraining loss: {:.4e}\tTesting loss: {:.4e}"
              .format(epoch, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                      loss_train, loss_test))
        with experiment.train():
            experiment.log_metric('Loss', loss_train)
        with experiment.test():
            experiment.log_metric('Loss', loss_test)
        saver.save(sess, "./saved_models/autoencoder-{}-final.ckpt".format(now))

        fig_compare, axes = plot_utils.plot_comparison(sess, data_test, X, output, hyperparams)
        fig_worst, axes = plot_utils.plot_worst(sess, data_train, X, output, hyperparams)
        experiment.log_figure(figure_name="comparison", figure=fig_compare)
        experiment.log_figure(figure_name="worst examples", figure=fig_worst)

        # log tensorflow graph and variables
        checkpoint_name = "./saved_models/autoencoder-{}-final.ckpt".format(now)
        experiment.log_asset(checkpoint_name + ".index")
        experiment.log_asset(checkpoint_name + ".meta")
        experiment.log_asset(checkpoint_name + ".data-00000-of-00001")


if __name__ == '__main__':
    experiment = Experiment(project_name="sweep-langmuir-ml", workspace="physicistphil")
    hyperparams = {'n_inputs': 500,
                   'scale': 0.0,  # no regularization
                   'learning_rate': 1e-3,
                   'momentum': 0.9,
                   'frac_train': 0.6,
                   'frac_test': 0.2,
                   'frac_valid': 0.2,
                   'batch_size': 256,
                   'steps': 100,
                   'seed': 42}
    experiment.add_tag("deep-7")
    experiment.log_parameters(hyperparams)
    train(experiment, hyperparams)
