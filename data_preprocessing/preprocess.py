from bapsflib import lapd
import numpy as np


def get_mirror_data(input_size):
    data_dir = "/home/phil/Desktop/sweeps/data/"

    f = [lapd.File(data_dir + "09_500G_flat_sweep_vf_correct2.hdf5"),
         lapd.File(data_dir + "10_500G_7to8_750Gother_sweep_vf.hdf5"),
         lapd.File(data_dir + "11_500G_7to8_1000Gother_sweep_vf.hdf5"),
         lapd.File(data_dir + "12_500G_7to8_1250Gother_sweep_vf.hdf5"),
         lapd.File(data_dir + "13_500G_7to8_1500Gother_sweep_vf.hdf5")]

    # TODO: find way to incorporate voltage curve into model

    time_len = f[0].read_data(2, 2, silent=True, shotnum=[1])['signal'].shape[1]
    stride = int(time_len / input_size)
    data = [_.read_data(2, 2, silent=True)['signal'][:, 0:input_size * stride:stride] for _ in f]
    dataset = np.concatenate((data), axis=0)

    return dataset
