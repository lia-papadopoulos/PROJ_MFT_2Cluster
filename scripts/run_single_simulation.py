


#%% STANDARD IMPORTS

import time
import numpy as np
import importlib
import matplotlib.pyplot as plt

# IMPORT SIMULATION FUNCTIONS

from src.simulation_setup import setup_baseline_parameters, set_initial_voltage, setup_stimulation
from src.simulation_tools.fcn_simulation_EIextInput import fcn_simulate_expSyn

# IMPORT NETWORK GENERATION FUNCTION
from src.make_networks.generate_network import generate_network
from src.make_networks.generate_network import get_network_population_info

# IMPORT MFT FUNCTION
from src.MFT_tools.MFT_solve import solveMFT_fixedInDeg_EI_net_rootEqs
from src.MFT_tools.MFT_solve import solveMFT_fixedInDeg_EI_net_dynEqs
from src.MFT_tools.MFT_clusteredEINetworks_tools import fcn_compare_J_C_mft_sim

# IMPORT ANALYSIS FUNCTIONS
from src.sim_analysis_tools import compute_firing_rates

# IMPORT USER SETTINGS
import userSettings as settings


#%% UNPACK SETTINGS FILE
sim_params_path = settings.sim_params_path
sim_params_name = settings.sim_params_name
mft_params_name = settings.mft_params_name
burnTime = 0.25
window_std = 25e-3
window_step = 1e-3


#%% SEEDS
stimClusters_seed = np.random.choice(10000)
stimNeurons_seed = np.random.choice(1000)
networkSeed = np.random.choice(10000)
initialVoltage_seed = np.random.choice(10000)


#%% LOAD PARAMETERS
sim_params_module = ( ('%s.%s') % (sim_params_path, sim_params_name) )
params = importlib.import_module(sim_params_module).params 

mft_params_module = ( ('%s.%s') % (sim_params_path, mft_params_name) )
m_params = importlib.import_module(mft_params_module).mft_params 

#%% SIMULATION RUN

# basic setup
setup_baseline_parameters(params)

# make network
W = generate_network(params, networkSeed)

# get network population info
get_network_population_info(params)

# set initial voltage
set_initial_voltage(params, initialVoltage_seed)

# setup stimulus
setup_stimulation(params, stimClusters_seed, stimNeurons_seed)
    

#%% RUN MFT (ROOT SOLVER)

results = solveMFT_fixedInDeg_EI_net_rootEqs(params, m_params)
results['nu_out']

#%% RUN MFT (DYNAMICAL EQUATIONS)

results = solveMFT_fixedInDeg_EI_net_dynEqs(params, m_params)
results['nu_out']

#%% RUN SIMULATION    

# start timing
t0 = time.time()

# save voltage
params.save_voltage = True
timePts, spikes, v, I_exc, I_inh, I_o = fcn_simulate_expSyn(params, W)

# end timing
tf = time.time()
print('sim time = %0.3f seconds' %(tf-t0))


#%% COMPUTE CLUSTER RATES FROM SIMULATIONS

# parameters
p = params.p
TF = params.TF
T0 = params.T0
popsizeE = params.popsizeE
popsizeI = params.popsizeI

# compute firing rate of each cell
bin_times, rateE, rateI = compute_firing_rates.fcn_compute_time_resolved_rate_gaussian(params, spikes, T0, TF, window_std, window_step)

# remove burn time
inds_keep = np.nonzero((bin_times >= T0 + burnTime) & (bin_times <= TF - burnTime))[0]
bin_times = bin_times[inds_keep]
rateE = rateE[:, inds_keep]
rateI = rateI[:, inds_keep]    

# compute population rates
rates_popE = compute_firing_rates.fcn_compute_clusterRates_vs_time(popsizeE, rateE)
rates_popI = compute_firing_rates.fcn_compute_clusterRates_vs_time(popsizeI, rateI)

# E cluster rates
rates_cluE = rates_popE[:p,:].copy()
rates_bgE = np.mean(rates_popE[p,:])
avg_cluRate_E = np.mean(rates_cluE, 1)

# I cluster rates
rates_cluI = rates_popI[:p,:].copy()
rates_bgI = np.mean(rates_popI[p,:])
avg_cluRate_I = np.mean(rates_cluI, 1)

# print results
print(avg_cluRate_E, rates_bgE)
print(avg_cluRate_I, rates_bgI)


#%% PLOT RASTER

plt.figure(figsize=(5.0,4))

indsE = np.nonzero(spikes[1,:] < params.N_e)[0]
indsI = np.nonzero(spikes[1,:] >= params.N_e)[0]

plt.plot(spikes[0,indsE],spikes[1,indsE], '.', markersize=0.5, color='navy')
plt.plot(spikes[0,indsI],spikes[1,indsI], '.', markersize=0.5, color='firebrick')
plt.yticks([])
plt.xlabel('time [s]')
plt.ylabel('neuron ID')
plt.tight_layout()


#%% PLOT NETWORK

plt.figure(figsize=(5.0,4))
plt.imshow(W,cmap='bwr_r')
plt.xlabel('neurons')
plt.ylabel('neurons')
plt.xticks([],[])
plt.yticks([],[])
plt.clim([-np.max(np.abs(W)),np.max(np.abs(W))])
plt.tight_layout()


#%% show plots

plt.show()


#%% COMPARE J MATS BETWEEN SIMULATION AND MFT

Jmat_mft = results['Jmat_rec']
Cmat_mft = results['Cmat_rec']

mft_sim_compare, Jmat_mft, Cmat_mft, Jmat_sim_reduced, Cmat_sim_reduced = fcn_compare_J_C_mft_sim(params, Jmat_mft, Cmat_mft, W)

print(mft_sim_compare)

# %%
