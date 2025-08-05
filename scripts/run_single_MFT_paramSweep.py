
#%% STANDARD IMPORTS

import numpy as np
import matplotlib.pyplot as plt
import importlib

# IMPORT SIMULATION FUNCTIONS
from src.simulation_setup import setup_baseline_parameters

# IMPORT MFT FUNCTION
from src.MFT_tools.MFT_paramSweep import fcn_sweep_high_to_low_rate
from src.MFT_tools.MFT_paramSweep import fcn_sweep_low_to_high_rate

# IMPORT USER SETTINGS
import userSettings as settings

#%% UNPACK SETTINGS FILE
sim_params_path = settings.sim_params_path
sim_params_name = settings.sim_params_name
mft_params_name = settings.mft_params_name
burnTime = 0.25
window_std = 25e-3
window_step = 1e-3

#%% LOAD PARAMETERS
sim_params_module = ( ('%s.%s') % (sim_params_path, sim_params_name) )
params = importlib.import_module(sim_params_module).params 

mft_params_module = ( ('%s.%s') % (sim_params_path, mft_params_name) )
m_params = importlib.import_module(mft_params_module).mft_params 

#%% BASIC SETUP

setup_baseline_parameters(params)

#%% RUN MFT

results_backwards = fcn_sweep_high_to_low_rate(params, m_params)
results_forwards = fcn_sweep_low_to_high_rate(params, m_params)


#%% PLOT RESULTS

plt.figure()

x = results_backwards['sweep_params_array_back'][0, :]
y = results_backwards['nu_e_backSweep'][0,:,0]
plt.plot(x, y, '-o', color='blue', linewidth=2, markersize=2, label='active back')

x = results_backwards['sweep_params_array_back'][0, :]
y = results_backwards['nu_e_backSweep'][1,:,0]
plt.plot(x, y, '-o', color='red', linewidth=2, markersize=2, label='inactive back')


x = results_forwards['sweep_params_array_for'][0, :]
y = results_forwards['nu_e_forSweep'][0,:,0]
plt.plot(x, y, '-o', color='gray', linewidth=2, markersize=2, label='active for')

x = results_forwards['sweep_params_array_for'][0, :]
y = results_forwards['nu_e_forSweep'][1,:,0]
plt.plot(x, y, '-o', color='black', linewidth=2, markersize=2, label='inactive for')

plt.xlabel('JPlusEE')
plt.ylabel('E cluster rate')
plt.legend()

# %%

plt.figure()

x = results_backwards['sweep_params_array_back'][0, :]
y = results_backwards['nu_i_backSweep'][0,:,0]
plt.plot(x, y, '-o', color='blue', linewidth=4, label='active back')

x = results_backwards['sweep_params_array_back'][0, :]
y = results_backwards['nu_i_backSweep'][1,:,0]
plt.plot(x, y, '-o', color='red', linewidth=4, label='inactive back')

plt.xlabel('JPlusEE')
plt.ylabel('I cluster rate')
plt.legend()