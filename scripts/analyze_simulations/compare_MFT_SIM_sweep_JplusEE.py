
#%% SETTING UP

import numpy as np
import os
import glob
import matplotlib.pyplot as plt

from src.sim_analysis_tools import compute_firing_rates
from src.sim_analysis_tools.fcn_load_simulations import fcn_load_simulations
from src.MFT_analysis_tools.fcn_load_mft_data import fcn_load_mft_sweep_JeePlus

sim_path = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_MFT_2Cluster/simulations/sweep_JplusEE/'
mft_path = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_MFT_2Cluster/mft/sweep_JplusEE/'
save_path = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_MFT_2Cluster/Figures/sweep_JplusEE/' 
sim_params_path = 'src.simulation_parameters'
sim_params_name = 'params1'
window_std = 20e-3
window_step = 1e-3
burnTime = 0.1
save_plots = True


#%% FOR PLOTTING RASTERS AND RATES

def plot_rasters_rates(sim_params, param_val, spikes, bin_times, rates_cluE, bgRate_E, rates_cluI, bgRate_I, save=False, save_path=''):
    
    N_e = sim_params.N_e
    popIndsE = sim_params.popIndsE
    popIndsI = sim_params.popIndsI + N_e
    sweep_param_name = sim_params.sweep_param1_name

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
    axs[0].set_title( ('%s = %0.3f') % (sweep_param_name, param_val))

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
        plt.savefig( ('%s%s_rasters_rates_%s%0.3f.png') % (save_path, sim_params_name, sweep_param_name, param_val) )


#%% FOR PLOTTING MFT AND SIM RATES
    
def plot_mft_sim_rates(sim_params, inputPop, \
                       JplusEE_backwards_mft, JplusEE_forwards_mft, sweep_JplusEE_sim , \
                       activeRate_E_backwards_mft, inactiveRate_E_backwards_mft, bgRate_E_backwards_mft, \
                       activeRate_E_forwards_mft, inactiveRate_E_forwards_mft, bgRate_E_forwards_mft, \
                       avg_activeRate_E_1Active, avg_inactiveRate_E_1Active, avg_bgRate_E_1Active, \
                       save=False, save_path=''):

    plt.figure()
    # backwards
    plt.plot(JplusEE_backwards_mft, activeRate_E_backwards_mft, '-', linewidth=5, color='lightseagreen', label= ('active %s MFT' % inputPop)) 
    plt.plot(JplusEE_backwards_mft, inactiveRate_E_backwards_mft, '-', linewidth=2, color='mediumpurple', label= ('inactive %s MFT' % inputPop))
    if np.isnan(avg_bgRate_E_1Active[0]) == False:
        plt.plot(JplusEE_backwards_mft, bgRate_E_backwards_mft, '-', color='gold', linewidth=5, label= ('bgr %s MFT' % inputPop))
    # forwards
    '''
    plt.plot(JplusEE_forwards_mft, activeRate_E_forwards_mft, '--', linewidth=2, color='gray', label='clu %s MFT (uniform)')
    plt.plot(JplusEE_forwards_mft, inactiveRate_E_forwards_mft, '--', linewidth=2, color='gray')
    if np.isnan(avg_bgRate_E_1Active[0]) == False:
        plt.plot(JplusEE_forwards_mft, bgRate_E_forwards_mft, '--', color='darkgray', linewidth=2, label='bgr %s MFT (uniform)')
    '''
    # sims
    plt.plot(sweep_JplusEE_sim, avg_activeRate_E_1Active, 'o', color='lightseagreen', label= ('active %s SIM' % inputPop)) 
    plt.plot(sweep_JplusEE_sim, avg_inactiveRate_E_1Active, 'o',color='mediumpurple', label= ('inactive %s SIM' % inputPop))
    if np.isnan(avg_bgRate_E_1Active[0]) == False:
        plt.plot(sweep_JplusEE_sim, avg_bgRate_E_1Active, 'o', color='gold', label= ('bgr %s SIM' % inputPop))
    plt.xlabel('JplusEE')
    plt.ylabel('population rates [spks/sec]')
    plt.legend()

    if save:
        plt.savefig( ('%s%s_%srates_mft_sim_sweep%s.pdf') % (save_path, sim_params_name, inputPop, sim_params.sweep_param1_name) )


#%% MAIN FUNCTION

#def main():
run=True
if run:

    ### MAKE DIRECTORY FOR FIGURES
    fig_path = ( ('%s/%s/') % (save_path, sim_params_name) )
    os.makedirs(fig_path, exist_ok=True)

    ### LOAD MFT DATA

    # name of mft file
    fname = ( ('%s_mft_sweep_JplusEE.h5') % (sim_params_name))
    full_path_to_file = ( ('%s%s') % (mft_path, fname) )

    # load mft info
    backwards_sweep_MFT, forwards_sweep_MFT = fcn_load_mft_sweep_JeePlus(full_path_to_file)

    # unpack info we'll want to plot
    JplusEE_backwards_mft = backwards_sweep_MFT.JplusEE_back
    
    activeRate_E_backwards_mft = backwards_sweep_MFT.nu_e_backSweep[0,:,0]
    inactiveRate_E_backwards_mft = backwards_sweep_MFT.nu_e_backSweep[1,:,0]
    bgRate_E_backwards_mft = backwards_sweep_MFT.nu_e_backSweep[-1,:,0]

    activeRate_I_backwards_mft = backwards_sweep_MFT.nu_i_backSweep[0,:,0]
    inactiveRate_I_backwards_mft = backwards_sweep_MFT.nu_i_backSweep[1,:,0]
    bgRate_I_backwards_mft = backwards_sweep_MFT.nu_i_backSweep[-1,:,0]

    JplusEE_forwards_mft = forwards_sweep_MFT.JplusEE_for
    
    activeRate_E_forwards_mft = forwards_sweep_MFT.nu_e_forSweep[0,:,0]
    inactiveRate_E_forwards_mft = forwards_sweep_MFT.nu_e_forSweep[1,:,0]
    bgRate_E_forwards_mft = forwards_sweep_MFT.nu_e_forSweep[-1,:,0]
    
    activeRate_I_forwards_mft = forwards_sweep_MFT.nu_i_forSweep[0,:,0]
    inactiveRate_I_forwards_mft = forwards_sweep_MFT.nu_i_forSweep[1,:,0]
    bgRate_I_forwards_mft = forwards_sweep_MFT.nu_i_forSweep[-1,:,0]


    ### LOAD SIMULATION DATA

    # load one simulation to get array lengths
    pattern = ('%s_sweep_JplusEE*_network0_stim0_trial0.h5' % sim_params_name)
    full_path_pattern = os.path.join(sim_path, pattern)
    files_in_directory = glob.glob(full_path_pattern)

    # name of simulation file
    full_path_to_file = files_in_directory[0]

    # load simulation info
    sim_params, _  = fcn_load_simulations(full_path_to_file)

    # get array of swept parameter
    sweep_JplusEE_sim = sim_params.sweep_param1_values

    avg_activeRate_E_1Active = np.zeros(len(sweep_JplusEE_sim))
    avg_inactiveRate_E_1Active = np.zeros(len(sweep_JplusEE_sim))
    avg_bgRate_E_1Active = np.zeros(len(sweep_JplusEE_sim))

    avg_activeRate_I_1Active = np.zeros(len(sweep_JplusEE_sim))
    avg_inactiveRate_I_1Active = np.zeros(len(sweep_JplusEE_sim))
    avg_bgRate_I_1Active = np.zeros(len(sweep_JplusEE_sim))

    for indParam, param in enumerate(sweep_JplusEE_sim):

        # name of simulation file
        fname = ( ('%s_sweep_JplusEE%0.3f_network0_stim0_trial0.h5') % (sim_params_name,param))
        full_path_to_file = ( ('%s%s') % (sim_path, fname) )

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
        plot_rasters_rates(sim_params, param, spikes, bin_times, rates_cluE, bgRate_E, rates_cluI, bgRate_I, save=save_plots, save_path=fig_path)

    # plot mft and sim comparison
    plot_mft_sim_rates(sim_params, 'E', \
                       JplusEE_backwards_mft, JplusEE_forwards_mft, sweep_JplusEE_sim , \
                       activeRate_E_backwards_mft, inactiveRate_E_backwards_mft, bgRate_E_backwards_mft, \
                       activeRate_E_forwards_mft, inactiveRate_E_forwards_mft, bgRate_E_forwards_mft, \
                       avg_activeRate_E_1Active, avg_inactiveRate_E_1Active, avg_bgRate_E_1Active, \
                       save=save_plots, save_path=fig_path)

    plot_mft_sim_rates(sim_params, 'I', \
                       JplusEE_backwards_mft, JplusEE_forwards_mft, sweep_JplusEE_sim , \
                       activeRate_I_backwards_mft, inactiveRate_I_backwards_mft, bgRate_I_backwards_mft, \
                       activeRate_I_forwards_mft, inactiveRate_I_forwards_mft, bgRate_I_forwards_mft, \
                       avg_activeRate_I_1Active, avg_inactiveRate_I_1Active, avg_bgRate_I_1Active, \
                       save=save_plots, save_path=fig_path)

#%% MAIN FUNCTION

if __name__=='__main__':

    main()


# %%
