from bapsflib import lapd
import numpy as np


# This gets the raw, unscaled data straight from the digitizer.
# Was used for initial testing of the autoencoder strategy.
def get_raw_mirror_data(input_size):
    data_dir = "/home/phil/Desktop/sweeps/data/"

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
