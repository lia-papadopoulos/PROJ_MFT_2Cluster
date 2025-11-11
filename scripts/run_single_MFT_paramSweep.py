

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

print(sim_params_name)

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

#%% EXCITATORY POPULATIONS

plt.figure()

x = results_backwards['sweep_params_array_back'][0, :]
y = results_backwards['nu_e_backSweep'][0,:,0]
plt.plot(x, y, '-o', color='dimgrey', linewidth=2, markersize=2, label='cluster solution')

x = results_backwards['sweep_params_array_back'][0, :]
y = results_backwards['nu_e_backSweep'][1,:,0]
plt.plot(x, y, '-o', color='dimgrey', linewidth=2, markersize=2)


x = results_forwards['sweep_params_array_for'][0, :]
y = results_forwards['nu_e_forSweep'][0,:,0]
plt.plot(x, y, '-o', color='darkgray', linewidth=2, markersize=2, label='uniform solution')

x = results_forwards['sweep_params_array_for'][0, :]
y = results_forwards['nu_e_forSweep'][1,:,0]
plt.plot(x, y, '-o', color='darkgray', linewidth=2, markersize=2)

plt.xlabel('swept parameters')
plt.ylabel('E cluster rate')
plt.legend()

#%% INHIBITORY POPULATIONS

plt.figure()

x = results_backwards['sweep_params_array_back'][0, :]
y = results_backwards['nu_i_backSweep'][0,:,0]
plt.plot(x, y, '-o', color='dimgray', linewidth=2, markersize=2, label='cluster solution')

x = results_backwards['sweep_params_array_back'][0, :]
y = results_backwards['nu_i_backSweep'][1,:,0]
plt.plot(x, y, '-o', color='dimgray', linewidth=2, markersize=2)


x = results_forwards['sweep_params_array_for'][0, :]
y = results_forwards['nu_i_forSweep'][0,:,0]
plt.plot(x, y, '-o', color='darkgray', linewidth=2, markersize=2, label='uniform solution')

x = results_forwards['sweep_params_array_for'][0, :]
y = results_forwards['nu_i_forSweep'][1,:,0]
plt.plot(x, y, '-o', color='darkgray', linewidth=2, markersize=2)

plt.xlabel('swept parameters')
plt.ylabel('I cluster rate')
plt.legend()

#%% STABILITY RESULTS

plt.figure()

x = results_backwards['sweep_params_array_back'][0, :]
y = results_backwards['largest_realPart_eigS_back'][:,0]
plt.plot(x, y, '-o', color='dimgray', linewidth=2, markersize=2, label='cluster solution')

x = results_forwards['sweep_params_array_for'][0, :]
y = results_forwards['largest_realPart_eigS_for'][:,0]
plt.plot(x, y, '-o', color='darkgray', linewidth=2, markersize=2, label='uniform solution')

x = [np.min(results_forwards['sweep_params_array_for'][0, :]), np.max(results_forwards['sweep_params_array_for'][0, :])]
y = [0,0]
plt.plot(x, y, color='purple', linewidth=2)

plt.xlabel('swept parameters')
plt.ylabel('largest $Re(\lambda)$ of stability matrix')
plt.legend()


#%% RATES WITH STABILITY INCLUDED

stable_sol_forwards = np.nonzero(results_forwards['largest_realPart_eigS_for'][:,0] < 0)[0]
unstable_sol_forwards = np.nonzero(results_forwards['largest_realPart_eigS_for'][:,0] >= 0)[0]

stable_sol_back = np.nonzero(results_backwards['largest_realPart_eigS_back'][:,0] < 0)[0]
unstable_sol_back = np.nonzero(results_backwards['largest_realPart_eigS_back'][:,0] >= 0)[0]

#### excitatory

plt.figure()


# backwards solution

x = results_backwards['sweep_params_array_back'][0, stable_sol_back]
y = results_backwards['nu_e_backSweep'][0,stable_sol_back,0]
plt.plot(x, y, '-', color='dimgray', linewidth=4, markersize=2)

x = results_backwards['sweep_params_array_back'][0, unstable_sol_back]
y = results_backwards['nu_e_backSweep'][0,unstable_sol_back,0]
plt.plot(x, y, '--', color='dimgray', linewidth=4, markersize=2)

x = results_backwards['sweep_params_array_back'][0, stable_sol_back]
y = results_backwards['nu_e_backSweep'][1,stable_sol_back,0]
plt.plot(x, y, '-', color='dimgray', linewidth=4, markersize=2)

x = results_backwards['sweep_params_array_back'][0, unstable_sol_back]
y = results_backwards['nu_e_backSweep'][1,unstable_sol_back,0]
plt.plot(x, y, '--', color='dimgray', linewidth=4, markersize=2)

# forwards solution

x = results_forwards['sweep_params_array_for'][0, stable_sol_forwards]
y = results_forwards['nu_e_forSweep'][0,stable_sol_forwards,0]
plt.plot(x, y, '-', color='darkgray', linewidth=4, markersize=2)

x = results_forwards['sweep_params_array_for'][0, unstable_sol_forwards]
y = results_forwards['nu_e_forSweep'][0,unstable_sol_forwards,0]
plt.plot(x, y, '--', color='darkgray', linewidth=4, markersize=2)

x = results_forwards['sweep_params_array_for'][0, stable_sol_forwards]
y = results_forwards['nu_e_forSweep'][1,stable_sol_forwards,0]
plt.plot(x, y, '-', color='darkgray', linewidth=4, markersize=2)

x = results_forwards['sweep_params_array_for'][0, unstable_sol_forwards]
y = results_forwards['nu_e_forSweep'][1,unstable_sol_forwards,0]
plt.plot(x, y, '--', color='darkgray', linewidth=4, markersize=2)


x = np.nan
y = np.nan
plt.plot(x, y, '-', color='dimgray', linewidth=3, label='cluster stable')
plt.plot(x, y, '--', color='dimgray', linewidth=3, label='cluster unstable')
plt.plot(x, y, '-', color='darkgray', linewidth=3, label='uniform stable')
plt.plot(x, y, '--', color='darkgray', linewidth=3, label='uniform unstable')

plt.legend()
plt.ylabel('excitatory rates')
plt.xlabel('swept parameter value')


#### inhibitory

plt.figure()


# backwards solution

x = results_backwards['sweep_params_array_back'][0, stable_sol_back]
y = results_backwards['nu_i_backSweep'][0,stable_sol_back,0]
plt.plot(x, y, '-', color='dimgray', linewidth=4, markersize=2)

x = results_backwards['sweep_params_array_back'][0, unstable_sol_back]
y = results_backwards['nu_i_backSweep'][0,unstable_sol_back,0]
plt.plot(x, y, '--', color='dimgray', linewidth=4, markersize=2)

x = results_backwards['sweep_params_array_back'][0, stable_sol_back]
y = results_backwards['nu_i_backSweep'][1,stable_sol_back,0]
plt.plot(x, y, '-', color='dimgray', linewidth=4, markersize=2)

x = results_backwards['sweep_params_array_back'][0, unstable_sol_back]
y = results_backwards['nu_i_backSweep'][1,unstable_sol_back,0]
plt.plot(x, y, '--', color='dimgray', linewidth=4, markersize=2)

# forwards solution

x = results_forwards['sweep_params_array_for'][0, stable_sol_forwards]
y = results_forwards['nu_i_forSweep'][0,stable_sol_forwards,0]
plt.plot(x, y, '-', color='darkgray', linewidth=4, markersize=2)

x = results_forwards['sweep_params_array_for'][0, unstable_sol_forwards]
y = results_forwards['nu_i_forSweep'][0,unstable_sol_forwards,0]
plt.plot(x, y, '--', color='darkgray', linewidth=4, markersize=2)

x = results_forwards['sweep_params_array_for'][0, stable_sol_forwards]
y = results_forwards['nu_i_forSweep'][1,stable_sol_forwards,0]
plt.plot(x, y, '-', color='darkgray', linewidth=4, markersize=2)

x = results_forwards['sweep_params_array_for'][0, unstable_sol_forwards]
y = results_forwards['nu_i_forSweep'][1,unstable_sol_forwards,0]
plt.plot(x, y, '--', color='darkgray', linewidth=4, markersize=2)

x = np.nan
y = np.nan
plt.plot(x, y, '-', color='dimgray', linewidth=3, label='cluster stable')
plt.plot(x, y, '--', color='dimgray', linewidth=3, label='cluster unstable')
plt.plot(x, y, '-', color='darkgray', linewidth=3, label='uniform stable')
plt.plot(x, y, '--', color='darkgray', linewidth=3, label='uniform unstable')

plt.legend()
plt.ylabel('inhibitory rates')
plt.xlabel('swept parameter value')

# %%
