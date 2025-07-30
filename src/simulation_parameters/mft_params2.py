
# basic imports
from types import SimpleNamespace
import numpy as np

# initialize parameters
mft_params = SimpleNamespace()

# set parameters
mft_params.nu_vec = np.array([50,0,0,15,15,15])

# JplusEE sweep
mft_params.JplusEE_sweep_mft = np.arange(15,22.05,0.05)

# number of active clusters to look for in solution
mft_params.n_active_clusters_sweep = np.array([1])

# high and low rates to begin at
mft_params.nu_clusterHigh_E = 50.
mft_params.nu_clusterLow_E = 0.
mft_params.nu_clusterHigh_I = 15.
mft_params.nu_clusterLow_I = 15.
mft_params.nu_uniform_E = 20.
mft_params.nu_uniform_I = 20.

# dynamical equations
mft_params.nSteps_MFT_DynEqs = 10000
mft_params.dt_MFT_DynEqs = 1e-4
mft_params.tau_e_MFT_DynEqs = 1e-3
mft_params.tau_i_MFT_DynEqs = 1e-3
mft_params.stopThresh_MFT_DynEqs = 1e-8
mft_params.plot_MFT_DynEqs = False