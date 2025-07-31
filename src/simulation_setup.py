# SET UP SIM PARAMETERS

import numpy as np
import sys


def setup_baseline_parameters(sim_params):

    compute_totalN(sim_params)
    update_Jplus_netType(sim_params) 
    compute_Js(sim_params)
    compute_Cs(sim_params)
    set_baseline_external_inputs_ei(sim_params)

    return None

def set_initial_voltage(sim_params, rng_seed):

    if rng_seed == 'random':

        seed = np.random.randint(0,10000)
        rng = np.random.default_rng(seed)
        
    else:

        rng = np.random.default_rng(rng_seed)

    iVe = rng.uniform(sim_params.Vr_e, sim_params.Vth_e, sim_params.N_e)
    iVi = rng.uniform(sim_params.Vr_i, sim_params.Vth_i, sim_params.N_i)
    sim_params.iV = np.append(iVe, iVi)

    return None

def setup_stimulation(sim_params, seed_stimClusters, seed_stimNeurons):

    set_max_stim_rate(sim_params)
    get_stimulated_clusters(sim_params, seed_stimClusters)
    get_stimulated_neurons(sim_params, seed_stimNeurons)

    return None


#%% SUB FUNCTIONS

def compute_totalN(sim_params):

    # total number of neurons
    sim_params.N = sim_params.N_e + sim_params.N_i

    return None

def update_Jplus_netType(sim_params):

    # if homogeneous networks
    if sim_params.net_type == 'hom':
        
        sim_params.JplusEE = 1.0       # EE intra-cluster potentiation factor
        sim_params.JplusII = 1.0       # II intra-cluster potentiation factor
        sim_params.JplusEI = 1.0       # EI intra-cluster potentiation factor
        sim_params.JplusIE = 1.0       # IE intra-cluster potentiation factor

    return None


def compute_Js(sim_params):

    sim_params.Jee = sim_params.jee/np.sqrt(sim_params.N)
    sim_params.Jie = sim_params.jie/np.sqrt(sim_params.N)
    sim_params.Jei = sim_params.jei/np.sqrt(sim_params.N)
    sim_params.Jii = sim_params.jii/np.sqrt(sim_params.N)
    sim_params.Jee_ext = sim_params.jee_ext/np.sqrt(sim_params.N)
    sim_params.Jie_ext = sim_params.jie_ext/np.sqrt(sim_params.N)
    sim_params.Jei_ext = sim_params.jei_ext/np.sqrt(sim_params.N)
    sim_params.Jii_ext = sim_params.jii_ext/np.sqrt(sim_params.N)

    return None

def compute_Cs(sim_params):

    # number of connections
    sim_params.Cee = sim_params.pee*sim_params.N_e
    sim_params.Cei = sim_params.pei*sim_params.N_i
    sim_params.Cii = sim_params.pii*sim_params.N_i
    sim_params.Cie = sim_params.pie*sim_params.N_e
    sim_params.Cee_ext = sim_params.N_e*sim_params.pext_ee 
    sim_params.Cie_ext = sim_params.N_e*sim_params.pext_ie 
    sim_params.Cei_ext = sim_params.N_i*sim_params.pext_ei 
    sim_params.Cii_ext = sim_params.N_i*sim_params.pext_ii

    return None


def set_baseline_external_inputs_ei(sim_params):

    sim_params.nu_ext_ee = sim_params.mean_nu_ext_ee*np.ones(sim_params.N_e)
    sim_params.nu_ext_ie = sim_params.mean_nu_ext_ie*np.ones(sim_params.N_i)
    sim_params.nu_ext_ei = sim_params.mean_nu_ext_ei*np.ones(sim_params.N_e)
    sim_params.nu_ext_ii = sim_params.mean_nu_ext_ii**np.ones(sim_params.N_i)

    return None


#%% CLUSTERS

# compute cluster ID of each neuron
def fcn_compute_cluster_assignments(sim_params):

    # unpack s_params
    popsizeE = sim_params.popsizeE
    popsizeI = sim_params.popsizeI

    # number of E and I neurons
    Ne = np.sum(popsizeE)
    Ni = np.sum(popsizeI)

    # initialize outputs
    Ecluster_ids = np.zeros(Ne)
    Icluster_ids = np.zeros(Ni)

    # population start and end indices [E]
    pops_start_end = np.append(0, np.cumsum(popsizeE))

    # number of populations
    npops = np.size(pops_start_end)-1

    # loop over populations
    for popInd in range(0,npops,1):

        # cluster start and end
        startID = pops_start_end[popInd]
        endID = pops_start_end[popInd+1]

        Ecluster_ids[startID:endID] = popInd

    # population start and end indices [I]
    pops_start_end = np.append(0, np.cumsum(popsizeI))

    # number of populations
    npops = np.size(pops_start_end)-1

    # loop over populations
    for popInd in range(0,npops,1):

        # cluster start and end
        startID = pops_start_end[popInd]
        endID = pops_start_end[popInd+1]

        Icluster_ids[startID:endID] = popInd

    return Ecluster_ids, Icluster_ids


#%% STIMULATION


# stimulation amplitude
def set_max_stim_rate(sim_params):

    sim_params.stimRate_E = sim_params.stim_rel_amp*sim_params.mean_nu_ext_ee
    sim_params.stimRate_I = sim_params.stim_rel_amp*sim_params.mean_nu_ext_ie

    return None


def get_stimulated_clusters(sim_params, random_seed):
    
    # unpack sim_params
    p = sim_params.p
    f_selectiveClus = sim_params.f_selectiveClus

    # set random number generator using the specified seed
    if random_seed == 'random':
        random_seed = np.random.choice(10000,1)
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng(random_seed)

    # get number of selective clusters
    n_selectiveClus = np.round(f_selectiveClus*p, 0).astype(int)

    # get selective cluster ids
    selectiveClusters = rng.choice(p, size=n_selectiveClus, replace=False)
            
    # update sim_params
    sim_params.selectiveClusters = selectiveClusters

    # return    
    return None


def get_stimulated_neurons(sim_params, random_seed):

    # set random number generator using the specified seed
    if random_seed == 'random':
        random_seed = np.random.choice(10000,1)
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng(random_seed)

    # boolean arrays that denote which neurons receive stimulus
    sim_params.stim_Ecells = np.zeros(sim_params.N_e)
    sim_params.stim_Icells = np.zeros(sim_params.N_i)

    # if homogeneous network
    if sim_params.net_type == 'hom':

        fracStim = sim_params.f_Ecells_target*sim_params.f_selectiveClus
        nStim_cells = np.round(sim_params.N_e*fracStim).astype(int)
        stim_cells = rng.choice(sim_params.N_e, nStim_cells, replace=False)
        sim_params.stim_Ecells[stim_cells] = True

        fracStim = sim_params.f_Icells_target*sim_params.f_selectiveClus
        nStim_cells = np.round(sim_params.N_i*fracStim).astype(int)
        stim_cells = rng.choice(sim_params.N_i, nStim_cells, replace=False)
        sim_params.stim_Icells[stim_cells] = True


    # else if clustered network
    elif sim_params.net_type == 'cluster':

        # get assignment of neurons to clusters
        Ecluster_inds, Icluster_inds = fcn_compute_cluster_assignments(sim_params)

        # loop over selective clusters
        for cluInd in sim_params.selectiveClusters:

            #---------- Ecells -----------#

            # cells in this cluster
            cells_in_clu = np.nonzero(Ecluster_inds == cluInd)[0]

            # number to select
            nstim = np.round(sim_params.f_Ecells_target*np.size(cells_in_clu),0).astype(int)

            # randomly select fraction of them
            stim_cells = rng.choice(cells_in_clu, \
                                    size = nstim, \
                                    replace=False)

            # save to sim_params
            sim_params.stim_Ecells[stim_cells] = True


            #---------- Icells -----------#

            # cells in this cluster
            cells_in_clu = np.nonzero(Icluster_inds == cluInd)[0]

            # number to select
            nstim = np.round(sim_params.f_Icells_target*np.size(cells_in_clu),0).astype(int)

            # randomly select fraction of them
            stim_cells = rng.choice(cells_in_clu, \
                                    size = nstim, \
                                    replace=False)

            # save to sim_params
            sim_params.stim_Icells[stim_cells] = True


    else:
        sys.exit('unknown network type')


    return None


#%% function to specify string of swept parameters and their values

def fcn_swept_param_name_val_str(sim_params, indSweep):
        
    sweep_param_str = ''

    for i in range(0, sim_params.n_sweepParams):

        key_to_param_name = ( ('sweep_param%d_name') % (i+1))
        paramName = vars(sim_params)[key_to_param_name]

        key_to_param_values = ( ('sweep_param%d_values') % (i+1))
        paramValue = vars(sim_params)[key_to_param_values][indSweep]

        if i == 0:
            sweep_param_str = ( ('%s%s%0.3f') % (sweep_param_str, paramName, paramValue))
        else:
            sweep_param_str = ( ('%s_%s%0.3f') % (sweep_param_str, paramName, paramValue))
        

    return sweep_param_str
