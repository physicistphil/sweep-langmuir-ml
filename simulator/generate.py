import numpy as np


def generate_basic_trace(n, Vp, Te, vsweep, S=2e-6):
    # Only generating based on density, floating potential, and temperature.
    # Isat and other effects are not considered.
    # Inputs are numpy arrays
    # Returns a 4D array of IV sweeps indexed by n, Vp, Te, and vsweep (in that order)

    # n = numpy array of density in m^-3
    # Vp = numpy array of plasma potential in V
    # Te = numpy array of temperatures in eV
    # vwseep = numpy array of sweep voltages in V
    # S = effective area of probe in m^2

    me = 9.109e-31
    e = 1.602e-19

    Te = Te * e  # convert to Joules

    # make grid from inputs
    n_grid, Vp_grid, Te_grid, vsweep_grid = np.meshgrid(n, Vp, Te, vsweep, indexing='ij')
    # electron saturation current (Amps / sqrt(eV))
    I_esat = S * n_grid * e / np.sqrt(2 * np.pi * me)
    # calculate the current
    current = I_esat * np.sqrt(Te_grid) * np.exp(-e * (Vp_grid - vsweep_grid) / Te_grid)
    # find where the bias voltage exceeds plasma potential and cap to I_esat
    condition = Vp_grid < vsweep_grid
    # cap the current when the bias voltage exceeds the plasma potential
    current[condition] = I_esat[condition] * np.sqrt(Te_grid)[condition]

    return current
