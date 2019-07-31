import numpy as np


def generate_basic_trace_from_grid(n, Vp, Te, vsweep, S=2e-6, return_grids=False):
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

    # Te = Te * e  # convert to Joules

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

    if not return_grids:
        return current
    else:
        return n_grid, Vp_grid, Te_grid, vsweep_grid, current


def generate_random_traces_from_array(ne_range, Vp_range, Te_range,
                                      vsweep_lower_range, vsweep_upper_range,
                                      hyperparams, size, S=2e-6):
    # n = numpy array of density range in m^-3
    # Vp = numpy array of plasma potential range in V
    # Te = numpy array of temperature range in J
    # vsweep_lower_range = numpy array of the range of the lower bound of the voltage curve (in V)
    # vsweep_upper_range = numpy array of the range of the upper bound of the voltage curve (in V)
    # hyperparams: hyperparameters of the model -- we just need the seed and n_inputs
    # size: how many traces to generate
    # S = effective area of probe in m^2

    n_inputs = hyperparams['n_inputs']

    # Physical constants (mks)
    me = 9.109e-31
    e = 1.602e-19

    # The amount of flat space before and after each voltage sweep.
    vsweep_flat_before_range = np.array([0, int(n_inputs * 0.4)])
    vsweep_flat_after_range = np.array([0, int(n_inputs * 0.4)])

    # Generate voltage sweep flat space parameters randomly within the range given.
    np.random.seed(hyperparams['seed'])
    vsweep_lower = np.random.uniform(vsweep_lower_range[0], vsweep_lower_range[1], size)
    np.random.seed(hyperparams['seed'] + 1)  # +1 on the seed to avoid duplicate random values
    vsweep_upper = np.random.uniform(vsweep_upper_range[0], vsweep_upper_range[1], size)
    np.random.seed(hyperparams['seed'])
    vsweep_flat_before = np.random.randint(vsweep_flat_before_range[0],
                                           vsweep_flat_before_range[1], size)
    np.random.seed(hyperparams['seed'] + 1)  # +1 on the seed to avoid duplicate random values
    vsweep_flat_after = np.random.randint(vsweep_flat_after_range[0],
                                          vsweep_flat_after_range[1], size)
    vsweep = np.ndarray(shape=(size, n_inputs))

    # Construct the voltage sweep from the flat space before, middle line, and flat space after.
    # Real-life Langmuir sweeps usually have these flat spots before and after the actual sweep.
    # We generate our sweep longer than we need and cut it shorter so that there's a chance of
    #   having two-sided, one-sided, or no-sided flat spots.
    for i in range(size):
        # Genearate lower flat curve.
        lower = np.full(vsweep_flat_before[i], vsweep_lower[i])
        # Generate the linear sweep.
        middle = np.linspace(vsweep_lower[i], vsweep_upper[i],
                             int(n_inputs * 1.2) - (vsweep_flat_before[i] + vsweep_flat_after[i]))
        # Generate the flat spot after the sweep.
        upper = np.full(vsweep_flat_after[i], vsweep_lower[i])
        # Generate the random index to cut down our vsweep to a length of n_inputs.
        randidx = np.random.randint(0, int(n_inputs * 0.2))
        vsweep[i] = np.concatenate((lower, middle, upper))[randidx:randidx + n_inputs]

    # Generate the density, plasma potential, and temperature from the given ranges.
    np.random.seed(hyperparams['seed'])
    ne = np.repeat(np.exp(np.random.uniform(np.log(ne_range[0]), np.log(ne_range[1]), (size, 1))),
                   n_inputs, axis=1)
    np.random.seed(hyperparams['seed'])
    Vp = np.repeat(np.random.uniform(Vp_range[0], Vp_range[1], (size, 1)), n_inputs, axis=1)
    np.random.seed(hyperparams['seed'])
    Te = np.repeat(np.random.uniform(Te_range[0], Te_range[1], (size, 1)), n_inputs, axis=1)

    # Electron saturation current. (Amps / sqrt(eV))
    I_esat = S * ne * e / np.sqrt(2 * np.pi * me)
    # Calculate the current.
    current = I_esat * np.sqrt(Te) * np.exp(-e * (Vp - vsweep) / Te)
    # Find where the bias voltage exceeds plasma potential.
    esat_condition = Vp < vsweep
    # Cap the current to I_esat.
    current[esat_condition] = I_esat[esat_condition] * np.sqrt(Te)[esat_condition]

    return ne[:, 0], Vp[:, 0], Te[:, 0], vsweep, current
