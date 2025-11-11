
# basic imports
import sys
import numpy as np

# helper functions
import src.MFT_tools.MFT_solve as MFT_solve
import src.MFT_tools.MFT_clusteredEINetworks_tools as MFT_clusteredEINetworks_tools

#%% fcn_sweep_backwards

def fcn_sweep_high_to_low_rate(sim_params, mft_params):
    
    '''
    sweep over parameters and compute MFT solution
    '''
        
    # number of clusters
    nClu = sim_params.p
    
    # parameter values to sweep over
    n_sweepParams = sim_params.n_sweepParams
    sweep_param1_values = mft_params.sweep_param1_values
    n_sweepValues = np.size(sweep_param1_values)

    # number of active clusters to look for in solution
    n_activeClusters_sweep = mft_params.n_active_clusters_sweep
    
    # high and low rates to begin at
    nu_clusterHigh_E = mft_params.nu_clusterHigh_E
    nu_clusterHigh_I = mft_params.nu_clusterHigh_I
    nu_clusterLow_E = mft_params.nu_clusterLow_E
    nu_clusterLow_I = mft_params.nu_clusterLow_I
    nu_uniform_E = mft_params.nu_uniform_E
    nu_uniform_I = mft_params.nu_uniform_I

    # need to reverse parameters?
    reverse_params_high_to_low_rate = mft_params.reverse_params_high_to_low_rate
    
    # number of E and I pops
    n_e_pops = MFT_clusteredEINetworks_tools.fcn_n_dynPops(nClu)
    n_i_pops = MFT_clusteredEINetworks_tools.fcn_n_dynPops(nClu)

    # sanity checks
    if np.any(n_activeClusters_sweep > nClu):
        sys.exit('# of active clusters cannot be larger than the number of clusters')
        
    # initialize backwards sweep quantities
    sweep_params_array_back = np.zeros((n_sweepParams, n_sweepValues))
    nu_e_back = np.zeros((n_e_pops, n_sweepValues, len(n_activeClusters_sweep)))
    nu_i_back = np.zeros((n_i_pops, n_sweepValues, len(n_activeClusters_sweep)))
    n_activeClustersE_back = np.zeros((n_sweepValues, len(n_activeClusters_sweep)), dtype=int)
    n_activeClustersI_back = np.zeros((n_sweepValues, len(n_activeClusters_sweep)), dtype=int)
    S = np.zeros((n_sweepValues, len(n_activeClusters_sweep)), dtype='object')
    largest_realPart_eigS_back = np.zeros((n_sweepValues, len(n_activeClusters_sweep)))
     
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
        for indSweep in range(0,n_sweepValues,1):
            
            for i in range(0, n_sweepParams):

                key_to_param_name = ( ('sweep_param%d_name') % (i+1) )
                paramName = vars(sim_params)[key_to_param_name]

                key_to_param_values = ( ('sweep_param%d_values') % (i+1) )

                if reverse_params_high_to_low_rate:
                    paramValue = np.flip(vars(mft_params)[key_to_param_values])[indSweep]
                else:
                    paramValue = vars(mft_params)[key_to_param_values][indSweep]

                setattr(sim_params, paramName, paramValue)

                sweep_params_array_back[i, indSweep] = paramValue
                        
            # if first Jee+ value, solve using dynamical equations, then root
            if indSweep == 0:
                mft_results = MFT_solve.solveMFT_fixedInDeg_EI_net_dynEqs(sim_params, mft_params)
                mft_params.nu_vec = mft_results['nu_out'].copy()
                mft_results = MFT_solve.solveMFT_fixedInDeg_EI_net_rootEqs(sim_params, mft_params)
            else:
                mft_results = MFT_solve.solveMFT_fixedInDeg_EI_net_rootEqs(sim_params, mft_params)

            # output firing rate
            nu_vec = mft_results['nu_out'].copy()

            # if no solution found, try with dynamical equations, then root
            if np.isnan(nu_vec[0]) == True:
                
                print('trying cluster fixed point w/ dynamical equations')
                    
                # set initial rate vector to high J guess
                mft_params.nu_vec = nu_vec_highJ
                    
                # run MFT
                mft_results = MFT_solve.solveMFT_fixedInDeg_EI_net_dynEqs(sim_params, mft_params)
                mft_params.nu_vec = mft_results['nu_out'].copy()
                mft_results = MFT_solve.solveMFT_fixedInDeg_EI_net_rootEqs(sim_params, mft_params)
                    
                # output rates
                nu_vec = mft_results['nu_out'].copy()
                
            # if no solution found, try uniform fixed point with dynamical equations, then root
            if np.isnan(nu_vec[0]) == True:
                
                print('trying uniform fixed point w/ dynamical equations')

                # set initial rate vector to lowJ guess
                mft_params.nu_vec = nu_vec_lowJ
                
                # run MFT
                mft_results = MFT_solve.solveMFT_fixedInDeg_EI_net_dynEqs(sim_params, mft_params)
                mft_params.nu_vec = mft_results['nu_out'].copy()
                mft_results = MFT_solve.solveMFT_fixedInDeg_EI_net_rootEqs(sim_params, mft_params)

                # output rates
                nu_vec = mft_results['nu_out'].copy()
            
            # if no solution found, exit program
            if np.isnan(nu_vec[0]) == True:
                sys.exit('ERROR: could not find solution')
            
            # check that we found the solution we're looking for
                
            # number of active E and I clusters in solution
            n_activeClustersE_back[indSweep, ind_nActive] = MFT_clusteredEINetworks_tools.fcn_check_num_activeClusters(nu_vec[:n_e_pops], nClu)
            n_activeClustersI_back[indSweep, ind_nActive] = MFT_clusteredEINetworks_tools.fcn_check_num_activeClusters(nu_vec[n_e_pops:], nClu)
            
            # do we have the correct number of active clusters?           
            if ( ((n_activeClustersE_back[indSweep, ind_nActive] == n_activeClusters) or (n_activeClustersE_back[indSweep, ind_nActive] == nClu)) == False ):
                print(nu_vec)
                print(indSweep, ind_nActive, n_activeClustersE_back[indSweep, ind_nActive])
                sys.exit('ERROR: solution does not have correct # of active E clusters') 
        
            # save solution 
            nu_e_back[:, indSweep, ind_nActive] = nu_vec[:n_e_pops].copy()
            nu_i_back[:, indSweep, ind_nActive] = nu_vec[n_e_pops:].copy()  

            # save stability
            largest_realPart_eigS_back[indSweep, ind_nActive] = np.nanmax(mft_results['realPart_eigvals_S'])
            S[indSweep, ind_nActive] = mft_results['S']
                
            # update initial guess at solution           
            mft_params.nu_vec = nu_vec.copy()
        
        # next value of n_activeClusters
        print('num active clusters = %d' % n_activeClusters_sweep[ind_nActive])

    # output results
    results = {}
    results['sweep_params_array_back'] = sweep_params_array_back
    results['nu_e_backSweep'] = nu_e_back
    results['nu_i_backSweep'] = nu_i_back
    results['n_activeClustersE_back'] = n_activeClustersE_back
    results['n_activeClustersI_back'] = n_activeClustersI_back
    results['largest_realPart_eigS_back'] = largest_realPart_eigS_back

    return results



#%% fcn_JeePlus_sweep_forwards

def fcn_sweep_low_to_high_rate(sim_params, mft_params):
    
    # number of clusters
    nClu = sim_params.p
    
    # parameter values to sweep over
    n_sweepParams = sim_params.n_sweepParams
    sweep_param1_values = mft_params.sweep_param1_values
    n_sweepValues = np.size(sweep_param1_values)

    # need to reverse parameters
    reverse_params_high_to_low_rate = mft_params.reverse_params_high_to_low_rate

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

    # initialize forwards sweep quantities
    sweep_params_array_for = np.zeros((n_sweepParams, n_sweepValues))
    nu_e_for = np.zeros((n_e_pops, n_sweepValues, len(n_activeClusters_sweep)))
    nu_i_for = np.zeros((n_i_pops, n_sweepValues, len(n_activeClusters_sweep)))
    n_activeClustersE_for = np.zeros((n_sweepValues, len(n_activeClusters_sweep)), dtype=int)
    n_activeClustersI_for = np.zeros((n_sweepValues, len(n_activeClusters_sweep)), dtype=int)
    largest_realPart_eigS_for = np.zeros((n_sweepValues, len(n_activeClusters_sweep)))

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
        for indSweep in range(0,n_sweepValues,1):
            
            for i in range(0, n_sweepParams):

                key_to_param_name = ( ('sweep_param%d_name') % (i+1) )
                paramName = vars(sim_params)[key_to_param_name]

                key_to_param_values = ( ('sweep_param%d_values') % (i+1) )

                if reverse_params_high_to_low_rate == False:
                    paramValue = np.flip(vars(mft_params)[key_to_param_values])[indSweep]
                else:
                    paramValue = vars(mft_params)[key_to_param_values][indSweep]

                setattr(sim_params, paramName, paramValue)

                sweep_params_array_for[i, indSweep] = paramValue
                        
            # if first swept value, solve using dynamical equations
            if indSweep == 0:
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
            n_activeClustersE_for[indSweep, ind_nActive] = MFT_clusteredEINetworks_tools.fcn_check_num_activeClusters(nu_vec[:n_e_pops], nClu)
            n_activeClustersI_for[indSweep, ind_nActive] = MFT_clusteredEINetworks_tools.fcn_check_num_activeClusters(nu_vec[n_e_pops:], nClu)
            
            # do we have the correct number of active clusters?           
            if ( ((n_activeClustersE_for[indSweep, ind_nActive] == n_activeClusters) or (n_activeClustersE_for[indSweep, ind_nActive] == nClu)) == False ):
                print(nu_vec)
                print(indSweep, ind_nActive, n_activeClustersE_for[indSweep, ind_nActive])
                sys.exit('ERROR: solution does not have correct # of active E clusters') 
        
            # save solution 
            nu_e_for[:,indSweep, ind_nActive] = nu_vec[:n_e_pops].copy()
            nu_i_for[:,indSweep, ind_nActive] = nu_vec[n_e_pops:].copy()  

            # save stability
            largest_realPart_eigS_for[indSweep, ind_nActive] = np.nanmax(mft_results['realPart_eigvals_S'])
            
            # update initial guess at solution           
            mft_params.nu_vec = nu_vec.copy()
    
        
        # next value of n_activeClusters
        print('num active clusters = %d' % n_activeClusters_sweep[ind_nActive])

    # output results
    results = {}
    results['sweep_params_array_for'] = sweep_params_array_for
    results['nu_e_forSweep'] = nu_e_for
    results['nu_i_forSweep'] = nu_i_for
    results['n_activeClustersE_for'] = n_activeClustersE_for
    results['n_activeClustersI_for'] = n_activeClustersI_for
    results['largest_realPart_eigS_for'] = largest_realPart_eigS_for

    return results

