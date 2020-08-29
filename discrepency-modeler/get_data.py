import numpy as np

# Custom tools from other directories
import sys
sys.path.append('../utilities')
import preprocess
import generate


# Get data from experiments and synthetic data if necessary.
def sample_datasets(hyperparams):
    print("Getting data...", end=" ")
    sys.stdout.flush()

    seed = hyperparams['seed']
    n_inputs = hyperparams['n_inputs']
    # Number of examples to sample from each dataset
    num_examples = hyperparams['num_examples']
    # Number of synthetic examples to sample from each dataset
    num_synthetic_examples = hyperparams['num_synthetic_examples']

    sweeps = []
    for i, data_file in enumerate(hyperparams['datasets']):
        temp_data = np.load("../../data_training/" + data_file + ".npz")['sweeps']
        np.random.seed(seed + i)
        np.random.shuffle(temp_data)
        temp_data = temp_data[0:temp_data.shape[0] if num_examples > temp_data.shape[0]
                              else num_examples]
        sweeps.append(temp_data)

    sweeps = np.concatenate(sweeps, axis=0)
    # Add 4 zeros after each sweep -- first zero is a flag indicating whether the following
    #   physical parameters (ne, Vp, Te) are included in the loss function calculation. They are
    #   not included for physical sweeps because they have not been analyzed yet.
    sweeps = np.concatenate([sweeps, np.zeros((sweeps.shape[0], 4))], axis=1)

    if num_synthetic_examples != 0:
        sweeps_synthetic = []
        for i, data_file in enumerate(hyperparams['datasets_synthetic']):
            temp_data = np.load("../../data_synthetic/" + data_file + ".npz")['sweeps']
            np.random.seed(seed + i + 1000)
            np.random.shuffle(temp_data)
            temp_data = temp_data[0:temp_data.shape[0]
                                  if num_synthetic_examples > temp_data.shape[0]
                                  else num_synthetic_examples]
            temp_data[:, 0:n_inputs * 2] = preprocess.add_real_noise(temp_data[:, 0:n_inputs * 2],
                                                                     hyperparams, epoch=0)
            sweeps_synthetic.append(temp_data)

            print("Num synthetic examples: ", num_synthetic_examples, "\ndata shape: ",
                  temp_data.shape[0])

        sweeps_synthetic = np.concatenate(sweeps_synthetic, axis=0)
        sweeps = np.concatenate([sweeps, sweeps_synthetic])
        del sweeps_synthetic

    # Find the voltage sweep and current means and peak-to-peaks so the model is easier to train.
    vsweep_mean = np.full(hyperparams['n_inputs'], np.mean(sweeps[:, 0:n_inputs]))
    vsweep_ptp = np.full(hyperparams['n_inputs'], np.ptp(sweeps[:, 0:n_inputs]))
    current_mean = np.full(hyperparams['n_inputs'], np.mean(sweeps[:, n_inputs:n_inputs * 2]))
    current_ptp = np.full(hyperparams['n_inputs'], np.ptp(sweeps[:, n_inputs:n_inputs * 2]))
    # Combine the two so we have a nice neat X, y, and scalings tuple returned by the function.
    data_mean = np.concatenate((vsweep_mean, current_mean))
    data_ptp = np.concatenate((vsweep_ptp, current_ptp))

    # Voltage and current sweeps are already concatenated.
    # Centering and scaling the input so that it's easier to train.
    sweeps[:, 0:n_inputs * 2] = (sweeps[:, 0:n_inputs * 2] - data_mean) / data_ptp
    data_train, data_test, data_valid = preprocess.shuffle_split_data(sweeps, hyperparams)
    print("Done.")

    return data_train, data_test, data_valid, data_mean, data_ptp


def generate_synthetic_dataset(hyperparams):
    print("Generating traces...", end=" ", flush=True)

    # Number of examples to generate. There are 3264 * 5 real ones from the mirror dataset.
    # size = hyperparams['num_examples']
    size = 2 ** 20

    ne_range = np.array([1e16, 1e18])
    Vp_range = np.array([0, 20])
    e = 1.602e-19  # Elementary charge
    Te_range = np.array([0.5, 10]) * e  # We're defining it in terms of eV because it's comfortable.
    S = 2e-6  # Probe area in m^2

    # Voltages used when collecting real sweeps are within this range.
    vsweep_lower_range = np.array([-50, -20])
    vsweep_upper_range = np.array([20, 60])

    ne, Vp, Te, vsweep, current \
        = generate.generate_random_traces_from_array(ne_range, Vp_range, Te_range,
                                                     vsweep_lower_range, vsweep_upper_range,
                                                     hyperparams, size, S=S)

    synthetic_data = np.concatenate([vsweep, current, np.ones((size, 1)),
                                     ne[:, np.newaxis], Vp[:, np.newaxis], Te[:, np.newaxis]],
                                    axis=1)
    np.savez("../../data_synthetic/16-18_0-20_0-5-10_-50--20_20-60", sweeps=synthetic_data)
