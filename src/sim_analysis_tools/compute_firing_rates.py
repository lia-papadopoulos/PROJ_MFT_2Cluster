
import numpy as np

#%%

def fcn_compute_time_resolved_rate_bins(sim_params, spikes, bin_width):

    '''
    COMPUTE FIRING RATE OF EACH NEURON BY BINNING SPIKES INTO NON-OVERLAPPING RECTANGULAR WINDOWS
    '''
    
    # number of E and I neurons
    N = sim_params.N
    Ne = sim_params.N_e

    # beginning and end times of simulation
    To = sim_params.T0
    Tf = sim_params.TF
        
    # spike times
    ts = spikes[0,:]  
    
    # neuron IDs
    neuron_IDs = spikes[1,:].astype(int)
        
    # spike count bins
    bins = np.arange(To, Tf+bin_width, bin_width)
    bin_times = bins[:-1] + bin_width/2
    n_bins = np.size(bin_times)
    
    # rate vs time for each cell
    rate = np.zeros((N, n_bins))
    
    # bin spike counts
    for cellInd in range(0, N):
        
        # spike time inds of this cell
        spike_inds_cell = np.nonzero(neuron_IDs == cellInd)[0]
        spike_times_cell = ts[spike_inds_cell]
        rate[cellInd, :], _ = np.histogram(spike_times_cell, bins)
        rate[cellInd, :] = rate[cellInd, :]/bin_width
        
        
    # separate E and I neurons
    rateE = rate[:Ne,:]
    rateI = rate[Ne:,:]
    
    # return
    return bin_times, rateE, rateI

#%%

def fcn_apply_gaussianKernel(t,ts,sigma):

    '''
    APPLY GAUSSIAN KERNEL TO SPIKES
    '''
    
    conv_spike = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-((t-ts)**2)/(2*sigma**2))
    
    return conv_spike

def fcn_compute_time_resolved_rate_gaussian(sim_params, spikes, To, Tf, \
                                            window_std, window_step):
    
    
    '''
    COMPUTE TIME-RESOLVED FIRING RATE OF EACH NEURON BY CONVOLVING SPIKE TRAIN WITH 
    GAUSSIAN KERNEL OF CERTAIN WIDTH
    '''
    
    # number of E and I neurons
    N = sim_params.N
    Ne = sim_params.N_e

    # times at which time resolved firing rate will be computed
    t = np.arange(To,Tf+window_step,window_step)
        
    # spike times
    ts = spikes[0,:]  
    n_spks = len(ts)
    
    # neuron IDs
    neuron_IDs = spikes[1,:].astype(int)
    
    # intitialize firing rate array
    rate = np.zeros((N,len(t)))
    
    # loop over all spike times
    for ts_ind in range(0,n_spks,1):
        
        # neuron id of current index
        n = neuron_IDs[ts_ind]
        
        # convolve spike with gaussian at all times t
        conv_spike = fcn_apply_gaussianKernel(t,ts[ts_ind],window_std)
        
        # update estimate of rate of neuron n at all times t by adding 
        # contribution from the current spike
        rate[n,:] = rate[n,:] + conv_spike
        
    # separate E and I neurons
    rateE = rate[:Ne,:]
    rateI = rate[Ne:,:]
    
    # return
    return t, rateE, rateI

        
#%%

def fcn_compute_clusterRates_vs_time(popSize, neuronRates):

    '''
    compute time-resolved firing rates of clusters (i.e. avg across all neurons in given cluster)
    includes background population if popSize includes background population
    '''
    
    # get cluster assignments
    cluLabels = 0
    cluLabels = np.append(cluLabels, np.cumsum(popSize))
    nClu = len(cluLabels)-1 
    
    # initialize cluster rates vs time
    rates_clu = np.zeros((nClu,np.size(neuronRates,1)))

    # average rates over neurons in the same cluster
    for cluInd in range(0,nClu,1):
    
        # cluster start and end
        a = cluLabels[cluInd]
        b = cluLabels[cluInd+1]

        # average rate
        rates_clu[cluInd,:] = np.mean(neuronRates[a:b,:],0)
        
    return rates_clu


#%%

def fcn_compute_clusterActivation(clu_rates, thresh):

    '''
    compute binarized array denoting cluster activation times
    '''
    
    # binarize cluster rates to determine whether active/inactive at a given time
    clu_binarized = (clu_rates > thresh)*1  
    
    return clu_binarized


def fcn_compute_num_activeClusters(clu_rates, thresh):

    '''
    compute number active clusters at each time point
    '''
    
    # binarize cluster rates to determine whether active/inactive at a given time
    clu_activity = (clu_rates > thresh)*1
    
    # sum across clusters to determine number of active clusters at each point in time
    num_active_clusters = (np.sum(clu_activity,0)).astype(int)
    
    return num_active_clusters


def fcn_compute_activeCluster_rates_givenBinarized(tRate, clu_rates, clu_binarized):
      
    '''
    compute average rate of active clusters at each time point given cluster activation array
    '''
    
    # list of active clusters
    ids_activeClus = []
    
    # average rate of active clusters
    avgRate_activeClus = np.zeros(len(tRate))
    
    # loop across time
    for tInd in range(0,len(tRate),1):
        
        ids = np.nonzero(clu_binarized[:,tInd]==1)[0]
        
        ids_activeClus.append(ids)
        
        # if active clusters
        if (np.size(ids) != 0):
            avgRate_activeClus[tInd] = np.mean(clu_rates[ids,tInd])
        # otherwise
        else:
            avgRate_activeClus[tInd] = np.nan
        
    return avgRate_activeClus


def fcn_compute_inactiveCluster_rates_givenBinarized(tRate, clu_rates, clu_binarized):
    
    '''
    compute average rate of inactive clusters at each time point given cluster activation array
    '''
    
    # list of inactive clusters
    ids_inactiveClus = []
    
    # average rate of active clusters
    avgRate_inactiveClus = np.zeros(len(tRate))
    
    # loop across time
    for tInd in range(0,len(tRate),1):
        
        ids = np.nonzero(clu_binarized[:,tInd]==0)[0]
        
        ids_inactiveClus.append(ids)
        
        # if inactive clusters
        if (np.size(ids) != 0):
            avgRate_inactiveClus[tInd] = np.mean(clu_rates[ids,tInd])
        # otherwise
        else:
            avgRate_inactiveClus[tInd] = np.nan
        
    return avgRate_inactiveClus


def fcn_compute_popRate_XClustersActive(avgRate_pop, num_active_clusters, maxActive):

    '''
    compute the average rate of a population (e.g. active clusters, inactive clusters, background population)
    conditioned on X clusters being simultaneously active
    INPUTS
        avgRate_pop = average rate of population of interest at each time point
                      for example, might be avg rate of active clusters at each time point
        num_active_clusters = vector of number of active clusters at each time point
        maxActive = maximum # active clusters to consider   
    '''
    
    # initialize
    clusterRate_XActive = np.zeros(maxActive+1)
    
    # loop over each choice of # simultaneously active clusters
    for nActive in range(0,maxActive+1,1):
        
        # time points where nActive clusters were active
        tInd = np.where(num_active_clusters == nActive)[0]
        
        # average rates of active clusters across those time points
        clusterRate_XActive[nActive] = np.mean(avgRate_pop[tInd])
        
        
    return clusterRate_XActive


