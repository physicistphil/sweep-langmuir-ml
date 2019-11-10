import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('/home/phil/Desktop/sweeps/sweep-langmuir-ml/simulator')
import generate


def autoencoder_plot_comparison(sess, data_test, X, output, hyperparams, y_in=None):
    size = data_test.shape[0]

    if y_in is None:
        output_test = output.eval(session=sess, feed_dict={X: data_test})
    else:
        # Chop off first half of series so that only traces are generated.
        output_test = output.eval(session=sess,
                                  feed_dict={X: data_test,
                                             y_in: np.zeros((size, 3))})

    # Reshape in case our data is 2d (which would be the case for a convolutional autoencoder)
    data_test = np.reshape(data_test, (size, -1))
    output_test = np.reshape(output_test, (size, -1))

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8), sharex=True)
    fig.suptitle('Comparison of test set and reconstruction')

    np.random.seed(hyperparams['seed'])
    randidx = np.random.randint(size, size=(3, 4))

    for x, y in np.ndindex((3, 4)):
        axes[x, y].plot(data_test[randidx[x, y]], label="Input")
        axes[x, y].plot(output_test[randidx[x, y]], label="Reconstruction")
        axes[x, y].set_title("Index {}".format(randidx[x, y]))

    axes[0, 0].legend()

    # fig.tight_layout()

    return fig, axes


def autoencoder_plot_worst(sess, data_train, X, output, hyperparams):
    output_train = output.eval(session=sess, feed_dict={X: data_train})

    # TODO: divide by range of examples to normalize the error before sorting
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


def phys_plot_comparison(sess, data_test, X, output, hyperparams):
    size = data_test.shape[0]

    output_test = output.eval(session=sess, feed_dict={X: data_test})

    # Reshape in case our data is 2d (which would be the case for a convolutional autoencoder)
    data_test = np.reshape(data_test, (size, -1))
    output_test = np.reshape(output_test, (size, -1))

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8), sharex=True)
    fig.suptitle('Comparison of test set and physics-based reconstruction')

    np.random.seed(hyperparams['seed'])
    randidx = np.random.randint(size, size=(3, 4))

    for x, y in np.ndindex((3, 4)):
        axes[x, y].plot(data_test[randidx[x, y], (size // 2):], label="Input")
        axes[x, y].plot(output_test[randidx[x, y]], label="Reconstruction")
        axes[x, y].set_title("Index {}".format(randidx[x, y]))

    axes[0, 0].legend()

    # fig.tight_layout()

    return fig, axes


# Compare plots of the actual curve vs the one inferred from the model.
def inferer_plot_comparison(sess, data_test, X, output, mean, diff, hyperparams):
    output_test = output.eval(session=sess, feed_dict={X: data_test}) * diff + mean
    # generated_trace = output_test
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8), sharex=True)
    fig.suptitle('Comparison of test set trace and derived trace')
    np.random.seed(hyperparams['seed'])
    randidx = np.random.randint(data_test.shape[0], size=(3, 4))
    # TODO: remove this hardcoded variable and use voltage sweep for inference
    vsweep = np.linspace(-30, 70, 500)

    for x, y in np.ndindex((3, 4)):
        ne = np.array([output_test[randidx[x, y], 0]])
        Vp = np.array([output_test[randidx[x, y], 1]])
        Te = np.array([output_test[randidx[x, y], 2]])
        # We subtract min to center the sweep at 0 again.
        axes[x, y].plot(data_test[randidx[x, y]] - np.min(data_test[randidx[x, y]]), label="Input")
        axes[x, y].plot(generate.generate_basic_trace_from_grid(ne, Vp, Te, vsweep)[0, 0, 0],
                        label="Derived")
        axes[x, y].set_title("Index {}".format(randidx[x, y]))
    axes[0, 0].legend()

    return fig, axes


# Same as infer_plot_comparison, but we compensate for scaled X and a given vsweep.
def inferer_plot_comparison_including_vsweep(sess, X, X_test, X_mean, X_ptp,
                                             output, y_mean, y_ptp, hyperparams):
    output_test = output.eval(session=sess, feed_dict={X: X_test}) * y_ptp + y_mean
    vsweep_unscaled = (X_test * X_ptp + X_mean)[:, 0:hyperparams['n_inputs']]
    current_unscaled = (X_test * X_ptp + X_mean)[:, hyperparams['n_inputs']:]

    # generated_trace = output_test
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8), sharex=True)
    fig.suptitle('Comparison of test set trace and derived trace')
    np.random.seed(hyperparams['seed'])
    randidx = np.random.randint(X_test.shape[0], size=(3, 4))

    for x, y in np.ndindex((3, 4)):
        ne = np.array([output_test[randidx[x, y], 0]])
        Vp = np.array([output_test[randidx[x, y], 1]])
        Te = np.array([output_test[randidx[x, y], 2]])

        axes[x, y].plot(current_unscaled[randidx[x, y]], label="Input")
        axes[x, y].plot(generate.generate_basic_trace_from_grid
                        (ne, Vp, Te, vsweep_unscaled[randidx[x, y]])[0, 0, 0], label="Derived")
        axes[x, y].set_title("Index {}".format(randidx[x, y]))
    axes[0, 0].legend()

    return fig, axes


def inferer_plot_quant_hist(sess, data_test, X, output, hyperparams):
    output_test = output.eval(session=sess, feed_dict={X: data_test})

    n_bins = 20
    fig, axes = plt.subplots(figsize=(6, 4))
    axes.hist(output_test[:, 0], bins=n_bins, alpha=0.5, edgecolor="black", label="ne")
    axes.hist(output_test[:, 1], bins=n_bins, alpha=0.5, edgecolor="black", label="Vp")
    axes.hist(output_test[:, 2], bins=n_bins, alpha=0.5, edgecolor="black", label="Te")
    axes.legend()

    return fig, axes


def inferer_plot_worst():
    pass
