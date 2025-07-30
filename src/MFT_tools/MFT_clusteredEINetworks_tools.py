
# basic imports
import numpy as np

# other tools
import src.make_networks.generate_network as generate_network


#%% NUMBER OF POPULATIONS GIVEN NUMBER OF CLUSTERS

def fcn_n_dynPops(p):

    n_dynPops = p+1
    return n_dynPops

#%% fcn_make_initial_rateVector

def fcn_make_initial_rateVector_popA(nu_high, nu_low, n_clusters, n_activeClusters):

    '''
    make initial rate vector for population A=E,I given:
    nu_high: guess for active cluster rate
    nu_low: guess for inactive cluster rate
    n_clusters: total number of clusters
    n_activeClusters: number of active clusters to look for
    '''

    nu_vec = np.zeros(n_clusters + 1)
    nu_vec[:n_activeClusters] = nu_high
    nu_vec[n_activeClusters:] = nu_low

    return nu_vec

#%% fcn_check_num_activeClusters

def fcn_check_num_activeClusters(nu_vec, n_clusters):

    n_activeClusters = np.size(np.nonzero( nu_vec[:n_clusters] >= np.max(nu_vec[:n_clusters])-1e-6 )[0])    
    return n_activeClusters

#%% get existing populations based on number of clusters and fraction of background neurons
def fcn_find_existing_pops(bgrA, p):

    nPops = fcn_n_dynPops(p)
    frac_Acells_perCluster = generate_network.frac_neurons_per_cluster(bgrA, p)

    Apops_exist = np.zeros(nPops)
    Apops_exist[:p] = (frac_Acells_perCluster > 0)*1
    Apops_exist[p] = (bgrA > 0)*1

    return Apops_exist

#%% COMPUTE WEIGHT AND DEGREE MATRICES FOR EACH NETWORK BLOCK

def fcn_compute_weight_degree_mats_AB(CAB, JAB, JplusAB, p, bgrA, bgrB, depress_interCluster):

    # number of populations
    nPops = fcn_n_dynPops(p)

    # initialize weight and degree matrices
    weightAB = np.zeros((nPops, nPops))
    degreeAB = np.zeros((nPops, nPops))

    # number of Apop and Bpop clusters
    n_Aclu = nPops - 1
    n_Bclu = nPops - 1    

    # compute synaptic depression factor
    JminusAB = generate_network.fcn_compute_depressFactors(depress_interCluster, p, bgrA, bgrB, JplusAB)

    # compute inter and intra cluster weights
    JAB_intra, JAB_inter = generate_network.fcn_compute_intra_inter_clusterWeightsAB(JAB, JplusAB, JminusAB)

    # compute fraction neurons/cluster
    frac_Bcells_perCluster = generate_network.frac_neurons_per_cluster(bgrB, p)


    # compute weight matrix

    # inputs to A clusters
    for i in range(n_Aclu):
        # from B clusters
        for j in range(n_Bclu):
            # if same cluster
            if j == i:
                weightAB[i,j] = JAB_intra 
            # if different clusters
            if j!=i:
                weightAB[i,j] = JAB_inter
        # from B background
        j = n_Bclu
        weightAB[i,j] = JAB_inter 

    # inputs to A background
    i = n_Aclu
    # from B clusters
    for j in range(n_Bclu):
        weightAB[i,j] = JAB_inter
    # from B background
    j = n_Bclu
    weightAB[i,j] = JAB 

    # compute degree matrix

    # inputs to A pops
    for i in range(n_Aclu + 1):
        # from B clusters
        for j in range(n_Bclu):
            degreeAB[i,j] = int(frac_Bcells_perCluster*CAB)
        # from B background
        j = n_Bclu
        degreeAB[i,j] = int(bgrB*CAB)


    return weightAB, degreeAB


def fcn_compute_weight_degree_mats_AXext(CAX_ext, JAX_ext, p):
    
    nPops = fcn_n_dynPops(p)

    weightAX_ext = np.zeros(nPops)
    degreeAX_ext = np.zeros(nPops)

    for i in range(0, nPops):
        weightAX_ext[i] = JAX_ext
        degreeAX_ext[i] = CAX_ext


    return weightAX_ext, degreeAX_ext


#%% CONCATENATE WEIGHT AND DEGREE MATRICES FROM ALL POPULATIONS

def fcn_concat_weight_mat(weightEE, weightIE, weightEI, weightII, weightEX_ext, weightIX_ext):

    recWeights_toE = np.hstack((weightEE, weightEI))
    recWeights_toI = np.hstack((weightIE, weightII))
    recWeight_matrix = np.vstack((recWeights_toE, recWeights_toI))

    extWeight_matrix = np.concatenate((weightEX_ext, weightIX_ext))

    return recWeight_matrix, extWeight_matrix

def fcn_concat_degree_mat(degreeEE, degreeIE, degreeEI, degreeII, degreeEX_ext, degreeIX_ext):

    recDegree_toE = np.hstack((degreeEE, degreeEI))
    recDegree_toI = np.hstack((degreeIE, degreeII))
    recDegree_matrix = np.vstack((recDegree_toE, recDegree_toI))

    extDegree_matrix = np.concatenate((degreeEX_ext, degreeIX_ext))

    return recDegree_matrix, extDegree_matrix


#%% COMPUTE TOTAL WEIGHT AND DEGREE MATRICES

def fcn_compute_final_weight_degree_mats(s_params):

    Cee = s_params.Cee 
    Cei = s_params.Cei 
    Cii = s_params.Cii 
    Cie = s_params.Cie 
    Cee_ext = s_params.Cee_ext 
    Cie_ext = s_params.Cie_ext 
    Jee = s_params.Jee 
    Jei = s_params.Jei 
    Jii = s_params.Jii 
    Jie = s_params.Jie 
    Jee_ext = s_params.Jee_ext        
    Jie_ext = s_params.Jie_ext 
    p = s_params.p
    bgrE = s_params.bgrE
    bgrI = s_params.bgrI
    JplusEE = s_params.JplusEE
    JplusEI = s_params.JplusEI    
    JplusIE = s_params.JplusIE
    JplusII = s_params.JplusII
    depress_interCluster = s_params.depress_interCluster

    # recurrent weight and degree matrices for each population
    weightEE, degreeEE = fcn_compute_weight_degree_mats_AB(Cee, Jee, JplusEE, p, bgrE, bgrE, depress_interCluster)
    weightIE, degreeIE = fcn_compute_weight_degree_mats_AB(Cie, Jie, JplusIE, p, bgrI, bgrE, depress_interCluster)
    weightEI, degreeEI = fcn_compute_weight_degree_mats_AB(Cei, Jei, JplusEI, p, bgrE, bgrI, depress_interCluster)
    weightII, degreeII = fcn_compute_weight_degree_mats_AB(Cii, Jii, JplusII, p, bgrI, bgrI, depress_interCluster)

    # external weight and degree matrices for each population
    weightEX_ext, degreeEX_ext = fcn_compute_weight_degree_mats_AXext(Cee_ext, Jee_ext, p)
    weightIX_ext, degreeIX_ext = fcn_compute_weight_degree_mats_AXext(Cie_ext, Jie_ext, p)

    # concatenate
    recWeightMat, extWeightMat = fcn_concat_weight_mat(weightEE, weightIE, weightEI, weightII, weightEX_ext, weightIX_ext)
    recDegreeMat, extDegreeMat = fcn_concat_degree_mat(degreeEE, degreeIE, degreeEI, degreeII, degreeEX_ext, degreeIX_ext)

    # return
    return recWeightMat, recDegreeMat, extWeightMat, extDegreeMat



#%% COMPARE J AND C MATRICES BETWEEN MFT AND SIMS

def fcn_compare_J_C_mft_sim(params, Jmat_mft, Cmat_mft, Wsim):

    p = params.p
    popIndsE = params.popIndsE
    popIndsI = params.popIndsI
    N_e = params.N_e
    bgrE = params.bgrE
    bgrI = params.bgrI

    Jmat_sim = Wsim.copy()
    Cmat_sim = (Wsim!=0)*1

    nPops = p + 1

    Jmat_sim_reduced = np.zeros((nPops*2, nPops*2))
    Cmat_sim_reduced = np.zeros((nPops*2, nPops*2))

    # E to E
    for i in range(0, nPops):
        for j in range(0, nPops):

            inds_i = popIndsE[i,:]
            inds_j = popIndsE[j,:]
            Ji = Jmat_sim[ inds_i[0]:inds_i[1], :  ]
            Jij = Ji[:, inds_j[0]:inds_j[1]]
            Ci = Cmat_sim[ inds_i[0]:inds_i[1], :  ]
            Cij = Ci[:, inds_j[0]:inds_j[1]]
            Jmat_sim_reduced[i,j] = np.mean(Jij[Jij!=0])
            Cmat_sim_reduced[i,j] = np.mean(np.sum(Cij, 1))

    # I to E
    for i in range(0, nPops):
        for j in range(0, nPops):

            inds_i = popIndsE[i,:]
            inds_j = popIndsI[j,:] + N_e
            Ji = Jmat_sim[ inds_i[0]:inds_i[1], :  ]
            Jij = Ji[:, inds_j[0]:inds_j[1]]

            Ci = Cmat_sim[ inds_i[0]:inds_i[1], :  ]
            Cij = Ci[:, inds_j[0]:inds_j[1]]

            Jmat_sim_reduced[i,j + nPops] = np.mean(Jij[Jij!=0])
            Cmat_sim_reduced[i,j + nPops] = np.mean(np.sum(Cij, 1))

    # E to I
    for i in range(0, nPops):
        for j in range(0, nPops):

            inds_i = popIndsI[i,:] + N_e
            inds_j = popIndsE[j,:] 
            Ji = Jmat_sim[ inds_i[0]:inds_i[1], :  ]
            Jij = Ji[:, inds_j[0]:inds_j[1]]
            Ci = Cmat_sim[ inds_i[0]:inds_i[1], :  ]
            Cij = Ci[:, inds_j[0]:inds_j[1]]
            Jmat_sim_reduced[i + nPops, j] = np.mean(Jij[Jij!=0])
            Cmat_sim_reduced[i + nPops, j] = np.mean(np.sum(Cij, 1))


    # I to I
    for i in range(0, nPops):
        for j in range(0, nPops):

            inds_i = popIndsI[i,:] + N_e
            inds_j = popIndsI[j,:] + N_e
            Ji = Jmat_sim[ inds_i[0]:inds_i[1], :  ]
            Jij = Ji[:, inds_j[0]:inds_j[1]]
            Ci = Cmat_sim[ inds_i[0]:inds_i[1], :  ]
            Cij = Ci[:, inds_j[0]:inds_j[1]]
            Jmat_sim_reduced[i + nPops, j + nPops] = np.mean(Jij[Jij!=0])
            Cmat_sim_reduced[i + nPops, j + nPops] = np.mean(np.sum(Cij, 1))


    Epops = fcn_find_existing_pops(bgrE, p)
    Epops_exist = np.nonzero(Epops)[0]

    Ipops = fcn_find_existing_pops(bgrI, p)
    Ipops_exist = nPops + np.nonzero(Ipops)[0]

    pops_exist = np.append(Epops_exist, Ipops_exist)

    mft_sim_C_compare = np.array([])
    mft_sim_J_compare = np.array([])

    for i in pops_exist:
        for j in pops_exist:

            Cmft_ij = Cmat_mft[i,j]
            Csim_ij = Cmat_sim_reduced[i,j]
            
            Jmft_ij = Jmat_mft[i,j]
            Jsim_ij = Jmat_sim_reduced[i,j]

            if Cmft_ij != Csim_ij:
                mft_sim_C_compare = np.append(mft_sim_C_compare, False)
            else:
                mft_sim_C_compare = np.append(mft_sim_C_compare, True)

            if np.round(Jmft_ij,10) != np.round(Jsim_ij,10):
                mft_sim_J_compare = np.append(mft_sim_J_compare, False)
            else:
                mft_sim_J_compare = np.append(mft_sim_J_compare, True)

    if ( (np.any(mft_sim_C_compare == False)) or (np.any(mft_sim_J_compare == False)) ):

        print('ERROR: MFT AND SIM DO NOT MATCH')
        return False, Jmat_mft, Cmat_mft, Jmat_sim_reduced, Cmat_sim_reduced

    else:

        print('SUCCESS!')
        return True, Jmat_mft, Cmat_mft, Jmat_sim_reduced, Cmat_sim_reduced
    
