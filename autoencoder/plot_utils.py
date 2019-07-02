import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_comparison(sess, data_test, X, output, hyperparams):
    output_test = output.eval(session=sess, feed_dict={X: data_test})

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8), sharex=True)
    fig.suptitle('Comparison of test set and reconstruction')

    np.random.seed(hyperparams['seed'])
    randidx = np.random.randint(data_test.shape[0], size=(3, 4))

    for x, y in np.ndindex((3, 4)):
        axes[x, y].plot(data_test[randidx[x, y]], label="Input")
        axes[x, y].plot(output_test[randidx[x, y]], label="Reconstruction")
        axes[x, y].set_title("Index {}".format(randidx[x, y]))

    axes[0, 0].legend()

    # fig.tight_layout()

    return fig, axes


def plot_worst(sess, data_train, X, output, hyperparams):
    output_train = output.eval(session=sess, feed_dict={X: data_train})

    diff_sq = np.sum(np.square(data_train - output_train), axis=1)
    # diff_var = np.var(np.square(data_train), axis=1)
    idx_sq = np.argsort(diff_sq)
    # idx_var = np.argsort(diff_var)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6), sharex=True)
    fig.suptitle('Worst performing training examples')

    for x, y in np.ndindex((2, 3)):
        axes[x, y].plot(data_train[idx_sq[- ((x + 3) * y + 1)]], label="Input")
        axes[x, y].plot(output_train[idx_sq[- ((x + 3) * y + 1)]], label="Reconstruction")
        axes[x, y].set_title("Index {}".format(idx_sq[- ((x + 3) * y + 1)]))

    axes[0, 0].legend()

    # fig.tight_layout()

    return fig, axes
