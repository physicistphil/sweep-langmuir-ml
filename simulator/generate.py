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
    # find where the bias voltage exceeds plasma potential
    condition = Vp_grid < vsweep_grid
    # cap the current to I_esat
    current[condition] = I_esat[condition] * np.sqrt(Te_grid)[condition]

    return current


# TODO: this is broken. Need to account for Ti in the meshgrid.
# Ti grid is assumed to be the same size as the Te grid.
# It may just be better to roll this into the function above.
def generate_trace(ne, Vp, Te, Ti, vsweep, gas='He', S=2e-6):
    if gas != 'he':
        print("Different gasses are not implemented yet. Using Helium.")
    # assuming helium fill gas
    ni = ne / 4

    current_e = generate_basic_trace(ne, Vp, Te, vsweep, S=2e-6)

    mi = 6.646e-27
    e = 1.602e-19

    Ti = Ti * e  # convert to Joules

    # make grid from inputs
    ni_grid, Vp_grid, Ti_grid, vsweep_grid = np.meshgrid(ni, Vp, Ti, vsweep, indexing='ij')
    # ion saturation current as Bohm current (assuming Ti << Te) (Amps / sqrt(eV))
    I_isat = -0.6 * S * ni_grid * e / np.sqrt(mi)
    # calculate the current
    current_i = I_isat * np.sqrt(Ti_grid) * np.exp(e * (Vp_grid - vsweep_grid) / Ti_grid)
    # find where the bias voltage is more negative than the plasma potential
    isat_condition = Vp_grid > vsweep_grid
    # cap current to I_isat
    current_i[isat_condition] = I_isat[isat_condition] * np.sqrt(Ti_grid)[isat_condition]

    return current_e + current_i
