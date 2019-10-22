from bapsflib import lapd
import numpy as np


# This gets the raw, unscaled data straight from the digitizer.
# Was used for initial testing of the autoencoder strategy.
def get_raw_mirror_data(input_size):
    data_dir = "../../data/"

    try:
        dataset = np.load(data_dir + 'data_concat')['dataset']
    except FileNotFoundError:
        f = [lapd.File(data_dir + "09_500G_flat_sweep_vf_correct2.hdf5"),
             lapd.File(data_dir + "10_500G_7to8_750Gother_sweep_vf.hdf5"),
             lapd.File(data_dir + "11_500G_7to8_1000Gother_sweep_vf.hdf5"),
             lapd.File(data_dir + "12_500G_7to8_1250Gother_sweep_vf.hdf5"),
             lapd.File(data_dir + "13_500G_7to8_1500Gother_sweep_vf.hdf5")]

        # TODO: find way to incorporate voltage curve into model

        time_len = f[0].read_data(2, 2, silent=True, shotnum=[1])['signal'].shape[1]
        stride = int(time_len / input_size)
        data = [_.read_data(2, 2, silent=True)['signal']
                [:, 0:input_size * stride:stride] for _ in f]
        dataset = np.concatenate((data), axis=0)
        np.savez(data_dir + 'data_concat', dataset=dataset)

    return dataset


def get_mirror_data(input_size):
    # Divide by -11 because we measured the current backwards across 11 ohms.
    # Current is mA.
    # Multiplier information can be found in the HDF5 file run descriptions.
    # This multiplier is only guaranteed to be correct for the mirror Langmuir sweeps that
    # were taken during the week of June 17th, 2018.
    return get_raw_mirror_data(input_size) / -11


def get_mirror_data_with_sweeps(input_size):
    # Multiply by 100 because that's what high-voltage probe was that was set for.
    # This only applies to mirror data taken during the week of June 17th, 2018.
    data_dir = "../../data/"

    try:
        dataset = np.load(data_dir + 'data_with_sweeps.npz')['dataset']
    except FileNotFoundError:
        f = [lapd.File(data_dir + "09_500G_flat_sweep_vf_correct2.hdf5"),
             lapd.File(data_dir + "10_500G_7to8_750Gother_sweep_vf.hdf5"),
             lapd.File(data_dir + "11_500G_7to8_1000Gother_sweep_vf.hdf5"),
             lapd.File(data_dir + "12_500G_7to8_1250Gother_sweep_vf.hdf5"),
             lapd.File(data_dir + "13_500G_7to8_1500Gother_sweep_vf.hdf5")]

        sweep_len = f[0].read_data(2, 1, silent=True, shotnum=[1])['signal'].shape[1]
        trace_len = f[0].read_data(2, 2, silent=True, shotnum=[1])['signal'].shape[1]
        assert sweep_len == trace_len, "Voltage and current traces not the same length"

        # Get the sweeps and traces from the data files.
        # Reduce the length of the data from 65536 to input_size (usually 500).
        stride = int(trace_len / input_size)
        data_sweep = np.concatenate([_.read_data(2, 1, silent=True)['signal']
                                     [:, 0:input_size * stride:stride] for _ in f], axis=0)
        data_trace = np.concatenate([_.read_data(2, 2, silent=True)['signal']
                                     [:, 0:input_size * stride:stride] for _ in f], axis=0)
        # Multiply the sweeps by 100 because it was measured through a x100 HV probe.
        data_sweep = 100 * data_sweep
        # Flip the traces (so it's consistent with the voltage sweep).
        data_trace = -data_trace
        # Remove the offset of the traces by averaging the first 10 points and subtracting it.
        data_trace -= np.mean(data_trace[:, 0:10], axis=1)[:, np.newaxis]
        # Remove the spiky bit after the sweep is turned off.
        data_trace[np.where(data_trace < -0.02)] = 0
        # Divide by the resistor value (11 ohms) to get the trace current.
        data_trace = data_trace / 11.0
        # Merge the sweep and trace together like we do for the synthetic sweeps.
        dataset = np.concatenate((data_sweep, data_trace), axis=1)
        # Cache the result on disk (NVME SSDs FTW) to save time.
        np.savez(data_dir + 'data_with_sweeps', dataset=dataset)

    return dataset


# Shuffle the data and split it into training, testing, and validation sets based on hyperparams
def shuffle_split_data(data, hyperparams):
    np.random.seed(hyperparams['seed'])
    np.random.shuffle(data)
    data_size = data.shape[0]
    data_train = data[0:int(data_size * hyperparams['frac_train']), :]
    data_test = data[int(data_size * hyperparams['frac_train']):
                     int(data_size * (hyperparams['frac_test'] + hyperparams['frac_train'])), :]
    data_valid = data[int(data_size * (hyperparams['frac_test'] + hyperparams['frac_train'])):, :]

    return data_train, data_test, data_valid


# Add a random offset to the data.
def add_offset(X, hyperparams, epoch=0):
    offset_scale = hyperparams['offset_scale'] * np.ptp(X[:, hyperparams['n_inputs']:], axis=1)
    # Add the current epoch so we get some variety over the entire training run.
    np.random.seed(hyperparams['seed'] + epoch)
    offsets = np.random.uniform(-offset_scale, offset_scale)
    X[:, hyperparams['n_inputs']:] += offsets[:, np.newaxis]

    return X


# Add noise to the data.
def add_noise(X, hyperparams, epoch=0):
    noise_scale = hyperparams['noise_scale'] * np.ptp(X[:, hyperparams['n_inputs']:], axis=1)
    # Add the current epoch so we get some variety over the entire training run.
    np.random.seed(hyperparams['seed'] + epoch)
    noise = np.random.normal(np.zeros((X.shape[0], hyperparams['n_inputs'])),
                             np.repeat(noise_scale[:, np.newaxis], hyperparams['n_inputs'], axis=1),
                             (X.shape[0], hyperparams['n_inputs']))
    X[:, hyperparams['n_inputs']:] += noise

    return X


# Get the phase in r + ic from an angle.
def phase(angle):
    return np.cos(angle) + 1.0j * np.sin(angle)


# Add noise derived from the fluctuations on the Langmuir probe sweep.
# Hopefully this makes the training examples a bit more realistic and improve accuracy.
# fft_abs was calculated beforehand from the first mirror sweep data (09).
# fft_abs.shape = (51, 64, 500) -- 51 positions, 64 shots per position.
def add_real_noise(X, hyperparams, epoch=0):
    noise_scale = hyperparams['noise_scale'] * np.ptp(X[:, hyperparams['n_inputs']:], axis=1)
    spectrum_path = "../../data/"
    fft_abs = np.load(spectrum_path + "fft_abs.npz")['fft_abs']
    fft_abs = np.mean(fft_abs, axis=1)[0][np.newaxis, :]  # Average over shots, r = 0 cm position.

    # Add the current epoch so we get some variety over the entire training run.
    seed = hyperparams['seed'] + epoch
    np.random.seed(seed)
    random_angle = np.random.uniform(0.0, 2.0 * np.pi, size=(X.shape[0], hyperparams['n_inputs']))
    # np.random.seed(seed)
    # random_shot = np.random.randint(0, 64, shape=X.shape[0])
    # Normalizing to values to have roughly unity peak-to-peak by dividing by 0.2
    noise = np.real(np.fft.ifft(fft_abs *
                                phase(random_angle))) / 0.1
    # Scale noise by a random number so there are some sweeps with noise and some without much.
    np.random.seed(seed)
    random_scalaing = np.random.uniform(0.0, 1.0, size=(X.shape[0], 1))
    X[:, hyperparams['n_inputs']:] += noise * noise_scale[:, np.newaxis] * random_scalaing

    return X
