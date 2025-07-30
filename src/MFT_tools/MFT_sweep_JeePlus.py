
# basic imports
import sys
import numpy as np

# helper functions
import src.MFT_tools.MFT_solve as MFT_solve
import src.MFT_tools.MFT_clusteredEINetworks_tools as MFT_clusteredEINetworks_tools

#%% fcn_JeePlus_sweep_backwards

def fcn_JeePlus_sweep_backwards(sim_params, mft_params):
    
    '''
    sweep over JeePlus (high to low values) and compute MFT solution
    '''
        
    # number of clusters
    nClu = sim_params.p
    
    # Jplus values to sweep over
    JplusEE_sweep_mft = mft_params.JplusEE_sweep_mft

    # number of active clusters to look for in solution
    n_activeClusters_sweep = mft_params.n_active_clusters_sweep
    
    # high and low rates to begin at
    nu_clusterHigh_E = mft_params.nu_clusterHigh_E
    nu_clusterHigh_I = mft_params.nu_clusterHigh_I
    nu_clusterLow_E = mft_params.nu_clusterLow_E
    nu_clusterLow_I = mft_params.nu_clusterLow_I
    nu_uniform_E = mft_params.nu_uniform_E
    nu_uniform_I = mft_params.nu_uniform_I
    
    # number of E and I pops
    n_e_pops = MFT_clusteredEINetworks_tools.fcn_n_dynPops(nClu)
    n_i_pops = MFT_clusteredEINetworks_tools.fcn_n_dynPops(nClu)

    # sanity checks
    if np.any(n_activeClusters_sweep > nClu):
        sys.exit('# of active clusters cannot be larger than the number of clusters')
        
    # initialize backwards sweep quantities
    JplusEE_back = np.flip(np.sort(JplusEE_sweep_mft))
    nu_e_back = np.zeros((n_e_pops, len(JplusEE_back), len(n_activeClusters_sweep)))
    nu_i_back = np.zeros((n_i_pops, len(JplusEE_back), len(n_activeClusters_sweep)))
    n_activeClustersE_back = np.zeros((len(JplusEE_back), len(n_activeClusters_sweep)), dtype=int)
    n_activeClustersI_back = np.zeros((len(JplusEE_back), len(n_activeClusters_sweep)), dtype=int)
        
    # loop over number of active clusters in solution
    for ind_nActive in range(0, len(n_activeClusters_sweep)):
        
        # number of active clusters in solution
        n_activeClusters = n_activeClusters_sweep[ind_nActive]
        mft_params.n_active_clusters = n_activeClusters

        # make initial rate vectors for high JeePlus (cluster states)
        nu_vec_e_highJ = MFT_clusteredEINetworks_tools.fcn_make_initial_rateVector_popA(nu_clusterHigh_E, nu_clusterLow_E, nClu, n_activeClusters)
        nu_vec_i_highJ = MFT_clusteredEINetworks_tools.fcn_make_initial_rateVector_popA(nu_clusterHigh_I, nu_clusterLow_I, nClu, n_activeClusters)
        
        # make initial rate vectors for low JeePlus (uniform states)
        nu_vec_e_lowJ = MFT_clusteredEINetworks_tools.fcn_make_initial_rateVector_popA(nu_uniform_E, nu_uniform_E, nClu, n_activeClusters)
        nu_vec_i_lowJ = MFT_clusteredEINetworks_tools.fcn_make_initial_rateVector_popA(nu_uniform_I, nu_uniform_I, nClu, n_activeClusters)
        
        # append E and I into single vector
        nu_vec_highJ = np.append(nu_vec_e_highJ, nu_vec_i_highJ)
        nu_vec_lowJ = np.append(nu_vec_e_lowJ, nu_vec_i_lowJ)
          
        # set initial rate vector for MFT calculation to high J guess
        mft_params.nu_vec = nu_vec_highJ
    
        # loop over Jee+
        for Jind in range(0,len(JplusEE_back),1):
            
            # update value of Jplus
            sim_params.JplusEE = JplusEE_back[Jind]
                        
            # if first Jee+ value, solve using dynamical equations
            if Jind == 0:
                mft_results = MFT_solve.solveMFT_fixedInDeg_EI_net_dynEqs(sim_params, mft_params)
            else:
                mft_results = MFT_solve.solveMFT_fixedInDeg_EI_net_rootEqs(sim_params, mft_params)

            # output firing rate
            nu_vec = mft_results['nu_out'].copy()

            # if no solution found, try with dynamical equations
            if np.isnan(nu_vec[0]) == True:
                
                print('trying cluster fixed point w/ dynamical equations')
                    
                # set initial rate vector to high J guess
                mft_params.nu_vec = nu_vec_highJ
                    
                # run MFT
                mft_results = MFT_solve.solveMFT_fixedInDeg_EI_net_dynEqs(sim_params, mft_params)
                    
                # output rates
                nu_vec = mft_results['nu_out'].copy()
                
            # if no solution found, try uniform fixed point with dynamical equations
            if np.isnan(nu_vec[0]) == True:
                
                print('trying uniform fixed point w/ dynamical equations')

                # set initial rate vector to lowJ guess
                mft_params.nu_vec = nu_vec_lowJ
                
                # run MFT
                mft_results = MFT_solve.solveMFT_fixedInDeg_EI_net_dynEqs(sim_params, mft_params)
            
                # output rates
                nu_vec = mft_results['nu_out'].copy()
            
            # if no solution found, exit program
            if np.isnan(nu_vec[0]) == True:
                sys.exit('ERROR: could not find solution')
            
            # check that we found the solution we're looking for
                
            # number of active E and I clusters in solution
            n_activeClustersE_back[Jind, ind_nActive] = MFT_clusteredEINetworks_tools.fcn_check_num_activeClusters(nu_vec[:n_e_pops], nClu)
            n_activeClustersI_back[Jind, ind_nActive] = MFT_clusteredEINetworks_tools.fcn_check_num_activeClusters(nu_vec[n_e_pops:], nClu)
            
            # do we have the correct number of active clusters?           
            if ( ((n_activeClustersE_back[Jind, ind_nActive] == n_activeClusters) or (n_activeClustersE_back[Jind, ind_nActive] == nClu)) == False ):
                print(nu_vec)
                print(Jind, ind_nActive, n_activeClustersE_back[Jind, ind_nActive])
                sys.exit('ERROR: solution does not have correct # of active E clusters') 
        
            # save solution 
            nu_e_back[:,Jind, ind_nActive] = nu_vec[:n_e_pops].copy()
            nu_i_back[:,Jind, ind_nActive] = nu_vec[n_e_pops:].copy()  
            
            # update initial guess at solution           
            mft_params.nu_vec = nu_vec.copy()
        
            # next value of Jplus        
            print('JplusEE = %0.3f' % JplusEE_back[Jind])
        
        # next value of n_activeClusters
        print('num active clusters = %d' % n_activeClusters_sweep[ind_nActive])

    # output results
    results = {}
    results['JplusEE_back'] = JplusEE_back
    results['nu_e_backSweep'] = nu_e_back
    results['nu_i_backSweep'] = nu_i_back
    results['n_activeClustersE_back'] = n_activeClustersE_back
    results['n_activeClustersI_back'] = n_activeClustersI_back

    return results



#%% fcn_JeePlus_sweep_forwards

def fcn_JeePlus_sweep_forwards(sim_params, mft_params):

    '''
    sweep over JeePlus (low to high values) and compute MFT solution
    '''
    
    # number of clusters
    nClu = sim_params.p
    
    # Jplus values to sweep over
    JplusEE_sweep_mft = mft_params.JplusEE_sweep_mft

    # number of active clusters to look for in solution
    n_activeClusters_sweep = mft_params.n_active_clusters_sweep
    
    # high and low rates to begin at
    nu_clusterHigh_E = mft_params.nu_clusterHigh_E
    nu_clusterHigh_I = mft_params.nu_clusterHigh_I
    nu_clusterLow_E = mft_params.nu_clusterLow_E
    nu_clusterLow_I = mft_params.nu_clusterLow_I
    nu_uniform_E = mft_params.nu_uniform_E
    nu_uniform_I = mft_params.nu_uniform_I
    
    # number of E and I pops
    n_e_pops = MFT_clusteredEINetworks_tools.fcn_n_dynPops(nClu)
    n_i_pops = MFT_clusteredEINetworks_tools.fcn_n_dynPops(nClu)

    # sanity checks
    if np.any(n_activeClusters_sweep > nClu):
        sys.exit('# of active clusters cannot be larger than the number of clusters')

    # initialize fowards sweep quantities
    JplusEE_for = np.sort(JplusEE_sweep_mft)
    nu_e_for= np.zeros((n_e_pops, len(JplusEE_for), len(n_activeClusters_sweep)))
    nu_i_for = np.zeros((n_i_pops, len(JplusEE_for), len(n_activeClusters_sweep)))
    n_activeClustersE_for = np.zeros((len(JplusEE_for), len(n_activeClusters_sweep)), dtype=int)
    n_activeClustersI_for = np.zeros((len(JplusEE_for), len(n_activeClusters_sweep)), dtype=int)

    # loop over number of active clusters in solution
    for ind_nActive in range(0, len(n_activeClusters_sweep)):
        
        # number of active clusters in solution
        n_activeClusters = n_activeClusters_sweep[ind_nActive]
        mft_params.n_active_clusters = n_activeClusters

        # make initial rate vectors for high JeePlus (cluster states)
        nu_vec_e_highJ = MFT_clusteredEINetworks_tools.fcn_make_initial_rateVector_popA(nu_clusterHigh_E, nu_clusterLow_E, nClu, n_activeClusters)
        nu_vec_i_highJ = MFT_clusteredEINetworks_tools.fcn_make_initial_rateVector_popA(nu_clusterHigh_I, nu_clusterLow_I, nClu, n_activeClusters)
        
        # make initial rate vectors for low JeePlus (uniform states)
        nu_vec_e_lowJ = MFT_clusteredEINetworks_tools.fcn_make_initial_rateVector_popA(nu_uniform_E, nu_uniform_E, nClu, n_activeClusters)
        nu_vec_i_lowJ = MFT_clusteredEINetworks_tools.fcn_make_initial_rateVector_popA(nu_uniform_I, nu_uniform_I, nClu, n_activeClusters)
        
        # append E and I into single vector
        nu_vec_highJ = np.append(nu_vec_e_highJ, nu_vec_i_highJ)
        nu_vec_lowJ = np.append(nu_vec_e_lowJ, nu_vec_i_lowJ)

        # set initial rate vector for MFT calculation to low J guess
        mft_params.nu_vec = nu_vec_lowJ

        # loop over Jee+
        for Jind in range(0,len(JplusEE_for),1):

            # update value of Jplus
            sim_params.JplusEE = JplusEE_for[Jind]
                        
            # if first Jee+ value, solve using dynamical equations
            if Jind == 0:
                mft_results = MFT_solve.solveMFT_fixedInDeg_EI_net_dynEqs(sim_params, mft_params)
            else:
                mft_results = MFT_solve.solveMFT_fixedInDeg_EI_net_rootEqs(sim_params, mft_params)

            # output firing rate
            nu_vec = mft_results['nu_out'].copy()

            # if no solution found, try with dynamical equations
            if np.isnan(nu_vec[0]) == True:
                
                print('trying uniform fixed point w/ dynamical equations')
                    
                # set initial rate vector to low J guess
                mft_params.nu_vec = nu_vec_lowJ
                    
                # run MFT
                mft_results = MFT_solve.solveMFT_fixedInDeg_EI_net_dynEqs(sim_params, mft_params)
                    
                # output rates
                nu_vec = mft_results['nu_out'].copy()
            
            # if no solution found, try cluster fixed point with dynamical equations
            if np.isnan(nu_vec[0]) == True:
                
                print('trying cluster fixed point w/ dynamical equations')

                # set initial rate vector to lowJ guess
                mft_params.nu_vec = nu_vec_highJ
                
                # run MFT
                mft_results = MFT_solve.solveMFT_fixedInDeg_EI_net_dynEqs(sim_params, mft_params)
            
                # output rates
                nu_vec = mft_results['nu_out'].copy()
    
            # if no solution found, exit program
            if np.isnan(nu_vec[0]) == True:
                sys.exit('ERROR: could not find solution')
            
            # check that we found the solution we're looking for
                
            # number of active E and I clusters in solution
            n_activeClustersE_for[Jind, ind_nActive] = MFT_clusteredEINetworks_tools.fcn_check_num_activeClusters(nu_vec[:n_e_pops], nClu)
            n_activeClustersI_for[Jind, ind_nActive] = MFT_clusteredEINetworks_tools.fcn_check_num_activeClusters(nu_vec[n_e_pops:], nClu)
            
            # do we have the correct number of active clusters?           
            if ( ((n_activeClustersE_for[Jind, ind_nActive] == n_activeClusters) or (n_activeClustersE_for[Jind, ind_nActive] == nClu)) == False ):
                print(nu_vec)
                print(Jind, ind_nActive, n_activeClustersE_for[Jind, ind_nActive])
                sys.exit('ERROR: solution does not have correct # of active E clusters') 
        
            # save solution 
            nu_e_for[:,Jind, ind_nActive] = nu_vec[:n_e_pops].copy()
            nu_i_for[:,Jind, ind_nActive] = nu_vec[n_e_pops:].copy()  
            
            # update initial guess at solution           
            mft_params.nu_vec = nu_vec.copy()
        
            # next value of Jplus        
            print('JplusEE = %0.3f' % JplusEE_for[Jind])
        
        # next value of n_activeClusters
        print('num active clusters = %d' % n_activeClusters_sweep[ind_nActive])

    # output results
    results = {}
    results['JplusEE_for'] = JplusEE_for
    results['nu_e_forSweep'] = nu_e_for
    results['nu_i_forSweep'] = nu_i_for
    results['n_activeClustersE_for'] = n_activeClustersE_for
    results['n_activeClustersI_for'] = n_activeClustersI_for

    return results

