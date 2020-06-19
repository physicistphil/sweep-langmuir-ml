import numpy as np

# Custom tools from other directories
import sys
sys.path.append('../utilities')
import preprocess
import generate


# Get data from real experiments (so far just from the mirror dataset).
def sample_datasets(hyperparams):
    print("Getting data...", end=" ")
    sys.stdout.flush()

    seed = hyperparams['seed']
    n_inputs = hyperparams['n_inputs']
    # Number of examples to sample from each dataset
    num_examples = hyperparams['num_examples']
    # Number of synthetic examples to sample from
    num_synthetic_examples = hyperparams['num_synthetic_examples']

    # Randomly sample num_examples of sweeps from each dataset
    mirror_data = preprocess.get_mirror_data_with_sweeps(n_inputs)
    # Shuffle the data so that we can easily sample without replacement.
    np.random.seed(seed)
    np.random.shuffle(mirror_data)
    mirror_data = mirror_data[0:16320 if num_examples > 16320 else num_examples]

    lapd_edge1_data = np.load("../../data_from_others/Gurleen_data/edge1.npz")['sweeps']
    np.random.seed(seed + 1)
    np.random.shuffle(lapd_edge1_data)
    lapd_edge1_data = lapd_edge1_data[0:72900 if num_examples > 72900 else num_examples]

    lapd_edge2_data = np.load("../../data_from_others/Gurleen_data/edge2.npz")['sweeps']
    np.random.seed(seed + 2)
    np.random.shuffle(lapd_edge2_data)
    lapd_edge2_data = lapd_edge2_data[0:78300 if num_examples > 78300 else num_examples]

    lapd_core_data = np.load("../../data_from_others/Gurleen_data/core.npz")['sweeps']
    np.random.seed(seed + 3)
    np.random.shuffle(lapd_core_data)
    lapd_core_data = lapd_core_data[0:311170 if num_examples > 311170 else num_examples]

    smpd_data = np.load("../../data_from_others/Kamil_data/01_xy.npz")['sweeps']
    # Multiply by 2.105 to normalzie to a probe size of 2e-6. "Best guess" for the probe in the
    #   SMPD was "0.9 - 1.0 mm^2", whatever that means.
    #smpd_data[:, n_inputs:] *= 2.105
    np.random.seed(seed + 4)
    np.random.shuffle(smpd_data)
    smpd_data = smpd_data[0:454410 if num_examples > 454410 else num_examples ]

    # Merge all the dataset samples into one big one.
    sweeps = np.concatenate([mirror_data, lapd_edge1_data, lapd_edge2_data,
                             lapd_core_data, smpd_data], axis=0)
    del mirror_data
    del lapd_edge1_data
    del lapd_edge2_data
    del lapd_core_data
    del smpd_data

    # Add 4 zeros after each sweep -- first zero is a flag indicating whether the following
    #   physical parameters (ne, Vp, Te) are included in the loss function calculation. They are
    #   not included for physical sweeps because they have not been analyzed yet.
    sweeps = np.concatenate([sweeps, np.zeros((sweeps.shape[0], 4))], axis=1)

    synthetic_data = np.load("../../data_synthetic/16-18_0-20_0-5-10_-50--20_20-60.npz")['sweeps']
    # Load in synthetic data (which already has the extra flag and physical parameters).
    np.random.seed(seed + 5)
    np.random.shuffle(synthetic_data)
    synthetic_data = synthetic_data[0:num_synthetic_examples]
    # Apply noise and offset
    # preprocess.add_offset(data_test, hyperparams, epoch=0)
    synthetic_data[:, 0:n_inputs * 2] = preprocess.add_real_noise(synthetic_data[:, 0:n_inputs * 2],
                                                                  hyperparams, epoch=0)

    sweeps = np.concatenate([sweeps, synthetic_data])
    del synthetic_data

    # Shuffle the datasets together.
    # np.random.seed(seed + 6)
    # np.random.shuffle(sweeps)

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
