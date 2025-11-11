
#%% SETTING UP

import numpy as np
import os
import glob
import importlib
import matplotlib.pyplot as plt

from src.simulation_setup import fcn_swept_param_name_val_str
from src.sim_analysis_tools import compute_firing_rates
from src.sim_analysis_tools.fcn_load_simulations import fcn_load_simulations
from src.MFT_analysis_tools.fcn_load_mft_data import fcn_load_mft_paramSweep

import userSettings as settings

sim_path = settings.sim_path
mft_path = settings.mft_path
save_path = settings.save_path
sim_params_path = settings.sim_params_path
sim_params_name = settings.sim_params_name
window_std = settings.window_std
window_step = settings.window_step
burnTime = settings.burnTime
save_plots = settings.save_plots


#%% REMOVE OLD FIGURES BEFORE CREATING NEW ONES

def fcn_remove_old_figures(directory_path):
    
    # Ensure the directory exists before attempting to remove files
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                try:
                    os.remove(item_path)
                    print(f"Removed file: {item_path}")
                except OSError as e:
                    print(f"Error removing file {item_path}: {e}")
    else:
        print(f"Directory not found or is not a directory: {directory_path}")

    return None


#%% NAME OF SWEPT PARAMETERS AS A STRING

def fcn_sweep_param_name(sim_params):

    sweep_param_str = ''

    for i in range(0, sim_params.n_sweepParams):

        key_to_param_name = ( ('sweep_param%d_name') % (i+1))
        paramName = vars(sim_params)[key_to_param_name]

        if i == 0:
            sweep_param_str = ( ('%s%s') % (sweep_param_str, paramName))
        else:
            sweep_param_str = ( ('%s_%s') % (sweep_param_str, paramName))

    return sweep_param_str

#%% FOR PLOTTING RASTERS AND RATES

def plot_rasters_rates(sim_params, sweep_param_str, spikes, bin_times, rates_cluE, bgRate_E, rates_cluI, bgRate_I, save=False, save_path=''):
    
    N_e = sim_params.N_e
    popIndsE = sim_params.popIndsE
    popIndsI = sim_params.popIndsI + N_e

    _, axs = plt.subplots(2, 1, sharex=True)

    indsE1 = np.nonzero( (spikes[1,:] >= popIndsE[0,0]) & (spikes[1,:] < popIndsE[0,1]) )[0]
    indsE2 = np.nonzero( (spikes[1,:] >= popIndsE[1,0]) & (spikes[1,:] < popIndsE[1,1]) )[0]
    indsEbg = np.nonzero( (spikes[1,:] >= popIndsE[2,0]) & (spikes[1,:] < popIndsE[2,1]) )[0]

    indsI1 = np.nonzero( (spikes[1,:] >= popIndsI[0,0]) & (spikes[1,:] < popIndsI[0,1]) )[0]
    indsI2 = np.nonzero( (spikes[1,:] >= popIndsI[1,0]) & (spikes[1,:] < popIndsI[1,1]) )[0]
    indsIbg = np.nonzero( (spikes[1,:] >= popIndsI[2,0]) & (spikes[1,:] < popIndsI[2,1]) )[0]

    axs[0].plot(spikes[0,indsE1],spikes[1,indsE1], '.', markersize=0.5, color='lightseagreen')
    axs[0].plot(spikes[0,indsE2],spikes[1,indsE2], '.', markersize=0.5, color='mediumpurple')
    axs[0].plot(spikes[0,indsEbg],spikes[1,indsEbg], '.', markersize=0.5, color='gold')

    axs[0].plot(spikes[0,indsI1],spikes[1,indsI1], '.', markersize=0.5, color='lightseagreen')
    axs[0].plot(spikes[0,indsI2],spikes[1,indsI2], '.', markersize=0.5, color='mediumpurple')
    axs[0].plot(spikes[0,indsIbg],spikes[1,indsIbg], '.', markersize=0.5, color='gold')

    axs[0].set_yticks([])
    axs[0].set_xlabel('time [s]')
    axs[0].set_ylabel('neurons')
    axs[0].set_title( ('%s') % (sweep_param_str))

    axs[1].plot(bin_times, rates_cluE[0,:], color='lightseagreen', label='Eclu 1')
    axs[1].plot(bin_times, rates_cluE[1,:], color='mediumpurple', label='Eclu 2')
    axs[1].plot(bin_times, bgRate_E, color='gold', label='Ebgr')

    axs[1].plot(bin_times, rates_cluI[0,:], '--', color='lightseagreen', label='Iclu 1')
    axs[1].plot(bin_times, rates_cluI[1,:], '--', color='mediumpurple', label='Iclu 2')
    axs[1].plot(bin_times, bgRate_I, '--', color='gold', label='Ibgr')
    axs[1].set_xlabel('time [s]')
    axs[1].set_ylabel('rates [sp/s]')
    axs[1].legend()

    if save:
        plt.savefig( ('%s%s_rasters_rates_%s.png') % (save_path, sim_params_name, sweep_param_str) )

    plt.close()

#%% FOR PLOTTING MFT AND SIM RATES
    
def plot_mft_sim_rates(sim_params, inputPop, \
                       sweep_params_array_back_mft, sweep_params_array_for_mft, sweep_JplusEE_sim , \
                       activeRate_E_backwards_mft, inactiveRate_E_backwards_mft, bgRate_E_backwards_mft, \
                       activeRate_E_forwards_mft, inactiveRate_E_forwards_mft, bgRate_E_forwards_mft, \
                       avg_activeRate_E_1Active, avg_inactiveRate_E_1Active, avg_bgRate_E_1Active, \
                       save=False, save_path=''):
    
    # check dimensions
    if sweep_params_array_back_mft.ndim == 1:
        x_back = sweep_params_array_back_mft.copy()
        x_for = sweep_params_array_for_mft.copy()

    else:
        x_back = sweep_params_array_back_mft[0,:].copy()
        x_for = sweep_params_array_for_mft[0,:].copy()

    # label for x axis
    x_label = fcn_sweep_param_name(sim_params)

    plt.figure()
    # backwards
    plt.plot(x_back, activeRate_E_backwards_mft, '-', linewidth=4, color='dimgray', label= ('%s cluster MFT' % inputPop)) 
    plt.plot(x_back, inactiveRate_E_backwards_mft, '-', linewidth=4, color='dimgray')
    if np.isnan(avg_bgRate_E_1Active[0]) == False:
        plt.plot(x_back, bgRate_E_backwards_mft, '-', color='lightsteelblue', linewidth=4, label= ('%s bgr MFT' % inputPop))
    # forwards
    plt.plot(x_for, activeRate_E_forwards_mft, '-', linewidth=4, color='darkgray', label=('%s uniform MFT' % inputPop))
    plt.plot(x_for, inactiveRate_E_forwards_mft, '-', linewidth=4, color='darkgray')
    if np.isnan(avg_bgRate_E_1Active[0]) == False:
        plt.plot(x_for, bgRate_E_forwards_mft, '-', color='lavender', linewidth=4, label=('%s bgr MFT' % inputPop))
    # sims
    plt.plot(sweep_JplusEE_sim, avg_activeRate_E_1Active, 'o', color='lightseagreen', label= ('%s active SIM' % inputPop)) 
    plt.plot(sweep_JplusEE_sim, avg_inactiveRate_E_1Active, 'o',color='mediumpurple', label= ('%s inactive SIM' % inputPop))
    if np.isnan(avg_bgRate_E_1Active[0]) == False:
        plt.plot(sweep_JplusEE_sim, avg_bgRate_E_1Active, 'o', color='gold', label= ('%s bgr SIM' % inputPop))
    plt.xlabel(x_label)
    plt.ylabel('population rates [spks/sec]')
    plt.legend()

    if save:
        plt.savefig( ('%s%s_%srates_mft_sim_sweep%s.pdf') % (save_path, sim_params_name, inputPop, x_label) )

    plt.close()


#%% FOR PLOTTING MFT AND SIM RATES
    
def plot_mft_sim_rates_stability(sim_params, inputPop, \
                       sweep_params_array_back_mft, sweep_params_array_for_mft, sweep_JplusEE_sim , \
                       activeRate_E_backwards_mft, inactiveRate_E_backwards_mft, bgRate_E_backwards_mft, \
                       activeRate_E_forwards_mft, inactiveRate_E_forwards_mft, bgRate_E_forwards_mft, \
                      largest_realPart_eigS_back, largest_realPart_eigS_for, \
                       avg_activeRate_E_1Active, avg_inactiveRate_E_1Active, avg_bgRate_E_1Active, \
                       save=False, save_path=''):
    
    # check dimensions
    if sweep_params_array_back_mft.ndim == 1:
        x_back = sweep_params_array_back_mft.copy()
        x_for = sweep_params_array_for_mft.copy()
    else:
        x_back = sweep_params_array_back_mft[0,:].copy()
        x_for = sweep_params_array_for_mft[0,:].copy()

    # label for x axis
    x_label = fcn_sweep_param_name(sim_params)

    # stability
    stable_sol_forwards = np.nonzero(largest_realPart_eigS_for < 0)[0]
    unstable_sol_forwards = np.nonzero(largest_realPart_eigS_for >= 0)[0]
    stable_sol_back = np.nonzero(largest_realPart_eigS_back < 0)[0]
    unstable_sol_back = np.nonzero(largest_realPart_eigS_back >= 0)[0]

    # largest real part of stability matrix eigenvalues

    plt.figure()

    y = largest_realPart_eigS_back
    plt.plot(x_back, y, '-o', color='dimgray', linewidth=2, markersize=2, label='cluster solution')

    y = largest_realPart_eigS_for
    plt.plot(x_for, y, '-o', color='darkgray', linewidth=2, markersize=2, label='uniform solution')

    x = [np.min(x_for), np.max(x_for)]
    y = [0,0]
    plt.plot(x, y, color='purple', linewidth=2)

    plt.xlabel(x_label)
    plt.ylabel('largest $Re(\lambda)$ of stability matrix')
    plt.legend()
    if save:
        plt.savefig( ('%s%s_largestRealpartEigStability_mft_sweep%s.pdf') % (save_path, sim_params_name, x_label) )

    plt.close()


    # rates with stability

    plt.figure()

    # backwards solution

    y = activeRate_E_backwards_mft[stable_sol_back]
    plt.plot(x_back[stable_sol_back], y, '-', color='dimgray', linewidth=4, markersize=2)

    y = activeRate_E_backwards_mft[unstable_sol_back]
    plt.plot(x_back[unstable_sol_back], y, '--', color='dimgray', linewidth=4, markersize=2)

    y = inactiveRate_E_backwards_mft[stable_sol_back]
    plt.plot(x_back[stable_sol_back], y, '-', color='dimgray', linewidth=4, markersize=2)

    y = inactiveRate_E_backwards_mft[unstable_sol_back]
    plt.plot(x_back[unstable_sol_back], y, '--', color='dimgray', linewidth=4, markersize=2)

    # forwards solution

    y = activeRate_E_forwards_mft[stable_sol_forwards]
    plt.plot(x_for[stable_sol_forwards], y, '-', color='darkgray', linewidth=4, markersize=2)

    y = activeRate_E_forwards_mft[unstable_sol_forwards]
    plt.plot(x_for[unstable_sol_forwards], y, '--', color='darkgray', linewidth=4, markersize=2)

    y = inactiveRate_E_forwards_mft[stable_sol_forwards]
    plt.plot(x_for[stable_sol_forwards], y, '-', color='darkgray', linewidth=4, markersize=2)

    y = inactiveRate_E_forwards_mft[unstable_sol_forwards]
    plt.plot(x_for[unstable_sol_forwards], y, '--', color='darkgray', linewidth=4, markersize=2)



    x = np.nan
    y = np.nan
    plt.plot(x, y, '-', color='dimgray', label=('stable MFT (cluster)'))
    plt.plot(x, y, '--', color='dimgray', label=('unstable MFT (cluster)'))
    plt.plot(x, y, '-', color='darkgray', label=('stable MFT (uniform)'))
    plt.plot(x, y, '--', color='darkgray', label=('unstable MFT (uniform)'))

    # sims
    plt.plot(sweep_JplusEE_sim, avg_activeRate_E_1Active, 'o', color='lightseagreen', label= ('active SIM')) 
    plt.plot(sweep_JplusEE_sim, avg_inactiveRate_E_1Active, 'o',color='mediumpurple', label= ('inactive SIM'))
    if np.isnan(avg_bgRate_E_1Active[0]) == False:
        plt.plot(sweep_JplusEE_sim, avg_bgRate_E_1Active, 'o', color='gold', label= ('bgr SIM'))
    plt.xlabel(x_label)
    plt.ylabel('%s population rates [spks/sec]' % inputPop)
    plt.legend()

    if save:
        plt.savefig( ('%s%s_%srates_mft_sim_sweep%s_withStability.pdf') % (save_path, sim_params_name, inputPop, x_label) )

    plt.close()

#%% MAIN FUNCTION

def main():
#run=True
#if run:

    ### MAKE DIRECTORY FOR FIGURES
    fig_path = ( ('%s/%s/') % (save_path, sim_params_name) )
    os.makedirs(fig_path, exist_ok=True)

    ### REMOVE ANY EXISTING FIGURES
    fcn_remove_old_figures(fig_path)

    # LOAD SIMULATION PARAMETERS
    sim_params_module = ( ('%s.%s') % (sim_params_path, sim_params_name) )
    params = importlib.import_module(sim_params_module).params 


    ### LOAD MFT DATA

    # name of swept parameters
    sweep_param_str = fcn_sweep_param_name(params)
    
    # name of mft file
    fname = ( ('%s_mft_sweep_%s.h5') % (sim_params_name, sweep_param_str))
    full_path_to_file = ( ('%s%s') % (mft_path, fname) )

    # load mft info
    backwards_sweep_MFT, forwards_sweep_MFT = fcn_load_mft_paramSweep(full_path_to_file)

    # unpack info we'll want to plot
    sweep_params_array_back_mft = backwards_sweep_MFT.sweep_params_array_back
    
    largest_realPart_eigS_back = backwards_sweep_MFT.largest_realPart_eigS_back[:,0]

    activeRate_E_backwards_mft = backwards_sweep_MFT.nu_e_backSweep[0,:,0]
    inactiveRate_E_backwards_mft = backwards_sweep_MFT.nu_e_backSweep[1,:,0]
    bgRate_E_backwards_mft = backwards_sweep_MFT.nu_e_backSweep[-1,:,0]

    activeRate_I_backwards_mft = backwards_sweep_MFT.nu_i_backSweep[0,:,0]
    inactiveRate_I_backwards_mft = backwards_sweep_MFT.nu_i_backSweep[1,:,0]
    bgRate_I_backwards_mft = backwards_sweep_MFT.nu_i_backSweep[-1,:,0]

    sweep_params_array_for_mft = forwards_sweep_MFT.sweep_params_array_for
    
    largest_realPart_eigS_for = forwards_sweep_MFT.largest_realPart_eigS_for[:,0]

    activeRate_E_forwards_mft = forwards_sweep_MFT.nu_e_forSweep[0,:,0]
    inactiveRate_E_forwards_mft = forwards_sweep_MFT.nu_e_forSweep[1,:,0]
    bgRate_E_forwards_mft = forwards_sweep_MFT.nu_e_forSweep[-1,:,0]
    
    activeRate_I_forwards_mft = forwards_sweep_MFT.nu_i_forSweep[0,:,0]
    inactiveRate_I_forwards_mft = forwards_sweep_MFT.nu_i_forSweep[1,:,0]
    bgRate_I_forwards_mft = forwards_sweep_MFT.nu_i_forSweep[-1,:,0]

    ### LOAD SIMULATION DATA

    # load one simulation to get array lengths
    pattern = ('%s_sweep*_network0_stim0_trial0.h5' % sim_params_name)
    full_path_pattern = os.path.join(sim_path, sim_params_name, pattern)
    files_in_directory = glob.glob(full_path_pattern)

    # name of simulation file
    full_path_to_file = files_in_directory[0]

    # load simulation info
    sim_params, _  = fcn_load_simulations(full_path_to_file)

    # get array of swept parameter
    sweep_param_values_sim = sim_params.sweep_param1_values

    # number of values in sweep
    nSweep = np.size(sweep_param_values_sim)

    avg_activeRate_E_1Active = np.zeros(nSweep)
    avg_inactiveRate_E_1Active = np.zeros(nSweep)
    avg_bgRate_E_1Active = np.zeros(nSweep)

    avg_activeRate_I_1Active = np.zeros(nSweep)
    avg_inactiveRate_I_1Active = np.zeros(nSweep)
    avg_bgRate_I_1Active = np.zeros(nSweep)

    for indParam, _ in enumerate(sweep_param_values_sim):

        # swept parameter string
        sweep_params_str = fcn_swept_param_name_val_str(sim_params, indParam)

        # name of simulation file
        fname = ( ('%s_sweep_%s_network0_stim0_trial0.h5') % (sim_params_name, sweep_params_str))
        full_path_to_file = ( ('%s%s/%s') % (sim_path, sim_params_name, fname) )

        # load simulation info
        sim_params, spikes  = fcn_load_simulations(full_path_to_file)

        # extract some info we need from sim_params
        p = sim_params.p
        TF = sim_params.TF
        T0 = sim_params.T0
        popsizeE = sim_params.popsizeE
        popsizeI = sim_params.popsizeI

        # compute firing rate of each cell
        bin_times, rateE, rateI = compute_firing_rates.fcn_compute_time_resolved_rate_gaussian(sim_params, spikes, T0, TF, window_std, window_step)

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
        avg_cluRate_E = np.mean(rates_cluE)
        bgRate_E = rates_popE[p,:].copy()
        # I cluster rates
        rates_cluI = rates_popI[:p,:].copy()
        avg_cluRate_I = np.mean(rates_cluI)
        bgRate_I = rates_popI[p,:].copy()

        # compute number of active E clusters at each time point
        num_activeClustersE = compute_firing_rates.fcn_compute_num_activeClusters(rates_cluE, avg_cluRate_E)
        num_activeClustersI = compute_firing_rates.fcn_compute_num_activeClusters(rates_cluI, avg_cluRate_I)

        # if background only
        if np.isnan(avg_cluRate_I):
            num_activeClustersI = num_activeClustersE.copy()

        # compute binarized rate of E clusters at eacth time point
        ratesBinarized_cluE = compute_firing_rates.fcn_compute_clusterActivation(rates_cluE, avg_cluRate_E)
        ratesBinarized_cluI = compute_firing_rates.fcn_compute_clusterActivation(rates_cluI, avg_cluRate_I)

        # compute average rate of active clusters at each time point
        avg_activeRate_E = compute_firing_rates.fcn_compute_activeCluster_rates_givenBinarized(bin_times, rates_cluE, ratesBinarized_cluE)
        avg_activeRate_I = compute_firing_rates.fcn_compute_activeCluster_rates_givenBinarized(bin_times, rates_cluI, ratesBinarized_cluI)

        # compute average rate of active clusters at each time point
        avg_inactiveRate_E = compute_firing_rates.fcn_compute_inactiveCluster_rates_givenBinarized(bin_times, rates_cluE, ratesBinarized_cluE)
        avg_inactiveRate_I = compute_firing_rates.fcn_compute_inactiveCluster_rates_givenBinarized(bin_times, rates_cluI, ratesBinarized_cluI)

        # compute average rate of active E clusters conditioned on X E clusters being active together
        avg_activeRate_E_XActive = compute_firing_rates.fcn_compute_popRate_XClustersActive(avg_activeRate_E, num_activeClustersE, p)
        avg_activeRate_I_XActive = compute_firing_rates.fcn_compute_popRate_XClustersActive(avg_activeRate_I, num_activeClustersI, p)

        # compute average rate of inactive E clusters conditioned on X E clusters being active together
        avg_inactiveRate_E_XActive = compute_firing_rates.fcn_compute_popRate_XClustersActive(avg_inactiveRate_E, num_activeClustersE, p)
        avg_inactiveRate_I_XActive = compute_firing_rates.fcn_compute_popRate_XClustersActive(avg_inactiveRate_I, num_activeClustersI, p)

        # compute average rate of background E conditioned on X E clusters active being active together
        avg_bgRate_E_XActive = compute_firing_rates.fcn_compute_popRate_XClustersActive(bgRate_E, num_activeClustersE, p)
        avg_bgRate_I_XActive = compute_firing_rates.fcn_compute_popRate_XClustersActive(bgRate_I, num_activeClustersI, p)

        # store active E rate conditioned on 1 active cluster
        avg_activeRate_E_1Active[indParam] = avg_activeRate_E_XActive[1]
        avg_activeRate_I_1Active[indParam] = avg_activeRate_I_XActive[1]

        # store inactive E rate conditioned on 1 active cluster
        avg_inactiveRate_E_1Active[indParam] = avg_inactiveRate_E_XActive[1]
        avg_inactiveRate_I_1Active[indParam] = avg_inactiveRate_I_XActive[1]

        # store average E rate conditioned on 1 active cluster
        avg_bgRate_E_1Active[indParam] = avg_bgRate_E_XActive[1]
        avg_bgRate_I_1Active[indParam] = avg_bgRate_I_XActive[1]

        # plot rasters and cluster rates
        plot_rasters_rates(sim_params, sweep_params_str, spikes, bin_times, rates_cluE, bgRate_E, rates_cluI, bgRate_I, save=save_plots, save_path=fig_path)

    # plot mft and sim comparison
    plot_mft_sim_rates(sim_params, 'E', \
                       sweep_params_array_back_mft, sweep_params_array_for_mft, sweep_param_values_sim , \
                       activeRate_E_backwards_mft, inactiveRate_E_backwards_mft, bgRate_E_backwards_mft, \
                       activeRate_E_forwards_mft, inactiveRate_E_forwards_mft, bgRate_E_forwards_mft, \
                       avg_activeRate_E_1Active, avg_inactiveRate_E_1Active, avg_bgRate_E_1Active, \
                       save=save_plots, save_path=fig_path)

    
    plot_mft_sim_rates(sim_params, 'I', \
                       sweep_params_array_back_mft, sweep_params_array_for_mft, sweep_param_values_sim , \
                       activeRate_I_backwards_mft, inactiveRate_I_backwards_mft, bgRate_I_backwards_mft, \
                       activeRate_I_forwards_mft, inactiveRate_I_forwards_mft, bgRate_I_forwards_mft, \
                       avg_activeRate_I_1Active, avg_inactiveRate_I_1Active, avg_bgRate_I_1Active, \
                       save=save_plots, save_path=fig_path)
    
    plot_mft_sim_rates_stability(sim_params, 'E', \
                       sweep_params_array_back_mft, sweep_params_array_for_mft, sweep_param_values_sim , \
                       activeRate_E_backwards_mft, inactiveRate_E_backwards_mft, bgRate_E_backwards_mft, \
                       activeRate_E_forwards_mft, inactiveRate_E_forwards_mft, bgRate_E_forwards_mft, \
                       largest_realPart_eigS_back, largest_realPart_eigS_for, \
                       avg_activeRate_E_1Active, avg_inactiveRate_E_1Active, avg_bgRate_E_1Active, \
                       save=save_plots, save_path=fig_path)

#%% MAIN FUNCTION

if __name__=='__main__':

    main()


# %%
