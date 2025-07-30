import numpy as np
import sys

#%% HELPER FUNCTIONS

#%%
def get_rng(seed):
    rng = np.random.default_rng(seed)
    return rng

#%% 
def check_num_clusters(net_type, p):
    if net_type == 'cluster':
        if p<1:
            sys.exit('must have at least one cluster if network_type = cluster')

#%%
def check_wholeNumber_background(N_alpha, bgr_alpha):

    leftovers = np.mod(N_alpha*bgr_alpha,1)
    if leftovers!=0.0:
        sys.exit('not a whole number of background neurons')

#%%
def check_wholeNumber_clustered(N_alpha, bgr_alpha, p):

    if ( abs((N_alpha*(1-bgr_alpha)/p)-round((N_alpha*(1-bgr_alpha)/p))) > 1e-8 ):
        sys.exit('not a whole number of clustered neurons')

#%%
def number_background_neurons(N_alpha, bgr_alpha):

    N_bgr_alpha = round(N_alpha*bgr_alpha)

    return N_bgr_alpha

#%%
def frac_neurons_per_cluster(bgrA, p):

    fA = (1-bgrA)/p  
    return fA

#%%
def fcn_compute_depressFactors(depress_interCluster, p, bgrA, bgrB, JplusAB):

    if depress_interCluster == False:

        JminusAB = 1.

    else:

        # fraction of neurons per cluster
        fA = frac_neurons_per_cluster(bgrA, p)
        fB = frac_neurons_per_cluster(bgrB, p)

        # if both are zero
        if ( (fA == 0) and (fB == 0) ):
            JminusAB = 1.
        else:
            # depression factor
            JminusAB = (fA + fB - p*fA*fB - fA*fB*JplusAB)/(fA + fB - p*fA*fB - fA*fB)

    return JminusAB


#%% COMPUTE INTRA AND INTER CLUSTER WEIGHT FACTORS

def fcn_compute_intra_inter_clusterWeightsAB(JAB, JplusAB, JminusAB):

    JAB_intra = JAB*JplusAB
    JAB_inter = JAB*JminusAB

    return JAB_intra, JAB_inter

#%%
def fcn_cells_in_each_pop(NA, bgrA, p, net_type):

    if net_type == 'hom':
        popsize = np.array([NA])
        cellInds_each_pop = np.array([0,NA])

    elif net_type == 'cluster':
        cellInds_each_pop = np.zeros((p+1,2), dtype=int)
        popsize = np.zeros(p+1, dtype=int)

        # if only background
        if bgrA == 1:
            popsize[p] = NA
            cellInds_each_pop[p,0] = 0
            cellInds_each_pop[p,1] = NA
        else:
            fA = frac_neurons_per_cluster(bgrA, p)
            Ncells_per_cluster = int(NA*fA)                # number units per cluster
            Ncells_background = number_background_neurons(NA, bgrA)  # number units in background
            popsize[:p] = Ncells_per_cluster
            popsize[p] = Ncells_background
            cusum_popsize = np.cumsum(np.append(0,popsize))
            for indPop in range(0,p+1):
                cellInds_each_pop[indPop,0] = cusum_popsize[indPop]
                cellInds_each_pop[indPop,1] = cusum_popsize[indPop+1]

    return popsize, cellInds_each_pop



#%%
def setup_network_conn(NA, NB, pAB, same_pops, rng):

    connMatAB = np.zeros((NA,NB))

    for i in range(0,NA):
        options = np.arange(NB)
        if same_pops:
            options = np.delete(options,i) # avoid self connections
        randConnections = rng.choice(options,int(pAB*NB),replace=False)
        connMatAB[i,randConnections] = 1

    return connMatAB

#%%
def setup_network_weights(NA, NB, pAB, jAB, jAB_in, jAB_out, p, bgrA, bgrB, same_pops, net_type, rng):


    # weight matrix
    weightMatAB = np.zeros((NA, NB))

    # cell indices in each population
    _, cellInds_Apops = fcn_cells_in_each_pop(NA, bgrA, p, net_type)
    _, cellInds_Bpops = fcn_cells_in_each_pop(NB, bgrB, p, net_type)

    # fraction neurons per cluster in population B
    fB = frac_neurons_per_cluster(bgrB, p)

    # cells in background
    inBackground_indsA = np.arange(cellInds_Apops[-1,0], cellInds_Apops[-1,1])
    inBackground_indsB = np.arange(cellInds_Bpops[-1,0], cellInds_Bpops[-1,1])

    # inputs to clusters
    for cluA in range(0,p):

        # cells in cluster of A group
        inCluster_indsA = np.arange(cellInds_Apops[cluA,0], cellInds_Apops[cluA,1])

        # loop over cellls
        for indCell in inCluster_indsA:

            # inputs to A cluster from B clusters
            for cluB in range(0,p):
                
                # cells in cluster of B group
                inCluster_indsB = np.arange(cellInds_Bpops[cluB,0], cellInds_Bpops[cluB,1])

                # if there are cells in this cluster
                if np.size(inCluster_indsB) >= 2:

                    # same cluster?
                    if cluB == cluA:
                        # don't allow self-connections
                        if same_pops:
                            removeInd = np.nonzero(inCluster_indsB==indCell)[0]
                            inCluster_indsB = np.delete(inCluster_indsB,removeInd) 
                        # make connections
                        randConnections = rng.choice(inCluster_indsB,int(pAB*NB*fB),replace=False)
                        weightMatAB[indCell,randConnections] = jAB_in 

                    # different clusters
                    else:
                        # make connections
                        randConnections = rng.choice(inCluster_indsB,int(pAB*NB*fB),replace=False)
                        weightMatAB[indCell,randConnections] = jAB_out 


            # inputs to A cluster from background
            if np.size(inBackground_indsB) >=1:
                randConnections = rng.choice(inBackground_indsB,int(pAB*NB*bgrB),replace=False)
                weightMatAB[indCell,randConnections] = jAB_out

    # inputs to background cells
    for indCell in inBackground_indsA:

        # from clusters
        for clu in range(0,p):
            inCluster_indsB = np.arange(cellInds_Bpops[clu,0], cellInds_Bpops[clu,1])
            if np.size(inCluster_indsB) >= 2:
                randConnections = rng.choice(inCluster_indsB,int(pAB*NB*fB),replace=False)
                weightMatAB[indCell,randConnections] = jAB_out

        # from background
        options = inBackground_indsB.copy()
        if np.size(options) >= 2:
            if same_pops:
                removeInd = np.nonzero(options==indCell)[0]
                options = np.delete(options,removeInd) 
            randConnections = rng.choice(options,int(pAB*NB*bgrB),replace=False)
            weightMatAB[indCell,randConnections] = jAB

    return weightMatAB


#%%
def combine_network_blocks(weightMatEE, weightMatEI, weightMatIE, weightMatII):

    JE = np.hstack((weightMatEE,weightMatEI))
    JI = np.hstack((weightMatIE,weightMatII))
    J = np.vstack((JE,JI))

    return J

#%% 

def check_diagonal(J):

    # extract diagonal
    matrix_diagonal = np.diag(J)

    if np.any(matrix_diagonal != 0):
        sys.exit('network should not have diagonal elements')


def check_inDegree(J, N_e):
    
    # binary network
    B=J!=0

    # in degree
    Kin = np.sum(B,1)

    # excitatory
    Kin_equal_E = np.all(Kin[:N_e] == Kin[0])

    # inhibitory
    Kin_equal_I = np.all(Kin[N_e:] == Kin[N_e])

    # check
    if ( (Kin_equal_E == False) or (Kin_equal_I == False) ):
        sys.exit('in degrees are not equal for all E/I neurons!')


#%% NETWORK GENERATION FUNCTION

def generate_network(sim_params, network_seed):

    # unpack parameters
    N_e = sim_params.N_e                                    # E neurons
    N_i = sim_params.N_i                                    # I neurons
    p = sim_params.p                                        # number of clusters
    bgrE = sim_params.bgrE                                  # fraction of background neurons (E)
    bgrI = sim_params.bgrI                                  # fraction of background neurons (I)
    depress_interCluster = sim_params.depress_interCluster  # whether or not to depress intercluster connections
    Jee = sim_params.Jee                                    # E-to-E synapses
    Jii = sim_params.Jii                                    # I-to-I synapses
    Jie = sim_params.Jie                                    # E-to-I synapses
    Jei = sim_params.Jei                                    # I-to-E synapses
    JplusEE = sim_params.JplusEE                            # E-to-E intra-cluster potentiation factor
    JplusII = sim_params.JplusII                            # I-to-I intra-cluster potentiation factor
    JplusEI = sim_params.JplusEI                            # E-to-I intra-cluster potentiation factor
    JplusIE = sim_params.JplusIE                            # I-to-E intra-cluster potentiation factor
    pee = sim_params.pee                                    # E-to-E connection prob
    pei = sim_params.pei                                    # I-to-E connection prob
    pii = sim_params.pii                                    # I-to-I connection prob
    pie = sim_params.pie                                    # E-to-I connection prob
    net_type = sim_params.net_type                  # homogeneous or clustered

    # random number generator for reproducibility
    if network_seed == 'random':
        rng = get_rng(np.random.randint(0,10000))
    else:
        rng = get_rng(network_seed)

    # if network_type == cluster, check that there's at least one cluster
    check_num_clusters(net_type, p)

    # check that there's a whole number of backgroud neurons in each popn
    check_wholeNumber_background(N_e, bgrE)
    check_wholeNumber_background(N_i, bgrI)

    # check that number of clusters is a factor of number available E and I
    # neurons to be clustered
    check_wholeNumber_clustered(N_e, bgrE, p)
    check_wholeNumber_clustered(N_i, bgrI, p)
   
    # compute depression factor for each type of weight (EE, EI, IE, II)
    JminusEE = fcn_compute_depressFactors(depress_interCluster, p, bgrE, bgrE, JplusEE)
    JminusIE = fcn_compute_depressFactors(depress_interCluster, p, bgrI, bgrE, JplusIE)
    JminusEI = fcn_compute_depressFactors(depress_interCluster, p, bgrE, bgrI, JplusEI)
    JminusII = fcn_compute_depressFactors(depress_interCluster, p, bgrI, bgrI, JplusII)

    # within and between cluster weights
    jEE_in, jEE_out = fcn_compute_intra_inter_clusterWeightsAB(Jee, JplusEE, JminusEE)
    jIE_in, jIE_out = fcn_compute_intra_inter_clusterWeightsAB(Jie, JplusIE, JminusIE)
    jEI_in, jEI_out = fcn_compute_intra_inter_clusterWeightsAB(Jei, JplusEI, JminusEI)
    jII_in, jII_out = fcn_compute_intra_inter_clusterWeightsAB(Jii, JplusII, JminusII)


    # check sign weights
    if ( (jEE_out < 0)  or (jIE_out < 0) ):
        sys.exit('got negative excitatory weights')
    
    if ( (jEI_out > 0)  or (jII_out > 0) ):
        sys.exit('got positive inhibitory weights')
        
    # setup different blocks of synaptic weight matrix
        
    # baseline weights
    connMatEE = setup_network_conn(N_e, N_e, pee, True, rng)
    connMatEI = setup_network_conn(N_e, N_i, pei, False, rng)
    connMatIE = setup_network_conn(N_i, N_e, pie, False, rng)
    connMatII = setup_network_conn(N_i, N_i, pii, True, rng)

    # if network type is homogeneous
    if net_type == 'hom':
        weightMatEE = connMatEE*Jee
        weightMatEI = connMatEI*Jei
        weightMatIE = connMatIE*Jie
        weightMatII = connMatII*Jii
    
    # otherwise, add in potentiation and depression
    else:
        weightMatEE = setup_network_weights(N_e, N_e, pee, Jee, jEE_in, jEE_out, p, bgrE, bgrE, True, net_type, rng)
        weightMatEI = setup_network_weights(N_e, N_i, pei, Jei, jEI_in, jEI_out, p, bgrE, bgrI, False, net_type, rng)
        weightMatIE = setup_network_weights(N_i, N_e, pie, Jie, jIE_in, jIE_out, p, bgrI, bgrE, False, net_type, rng)
        weightMatII = setup_network_weights(N_i, N_i, pii, Jii, jII_in, jII_out, p, bgrI, bgrI, True, net_type, rng)
    
    # put each block of the network together      
    J = combine_network_blocks(weightMatEE, weightMatEI, weightMatIE, weightMatII)


    # checks
    check_diagonal(J)
    check_inDegree(J, N_e)
        
    #-------------------------------------------------------------------------
    # RETURN
    #------------------------------------------------------------------------- 
    return J
    

#%% SET POPULATION SIZES AND VECTORS THAT STORE WHAT NEURONS ARE IN WHAT POPULATIONS

def get_network_population_info(sim_params):

    # unload sim_params
    N_e = sim_params.N_e
    N_i = sim_params.N_i
    p = sim_params.p
    bgrE = sim_params.bgrE
    bgrI = sim_params.bgrI
    net_type = sim_params.net_type

    # output population size vectors
    popsizeE, popIndsE = fcn_cells_in_each_pop(N_e, bgrE, p, net_type)
    popsizeI, popIndsI = fcn_cells_in_each_pop(N_i, bgrI, p, net_type)

    # update sim_params
    sim_params.popsizeE = popsizeE
    sim_params.popsizeI = popsizeI
    sim_params.popIndsE = popIndsE
    sim_params.popIndsI = popIndsI

    return None
