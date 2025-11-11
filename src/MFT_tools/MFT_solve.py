
# basic imports
import sys
import numpy as np

# other tools
import src.MFT_tools.MFT_basic_tools as MFT_basic_tools
import src.MFT_tools.MFT_clusteredEINetworks_tools as MFT_clusteredEINetworks_tools

#%% MAIN FUNCTION TO COMPUTE MFT SOLUTION (USING ROOT FINDING)

def solveMFT_fixedInDeg_EI_net_rootEqs(s_params, m_params):    
    
    # load simulation parameters       
    tau_r = s_params.tau_r              # refractory period
    tau_m_e = s_params.tau_m_e          # membrane time constant E
    tau_m_i = s_params.tau_m_i          # membrane time constant I
    tau_s_e = s_params.tau_s_e          # synaptic time constant E
    tau_s_i = s_params.tau_s_i          # synaptic time constant I
    Vr_e = s_params.Vr_e                # reset potential E
    Vr_i = s_params.Vr_i                # reset potential E
    Vth_e = s_params.Vth_e              # threshold potential E
    Vth_i = s_params.Vth_i              # threshold potential I
    nu_ext_ee = s_params.mean_nu_ext_ee
    nu_ext_ie = s_params.mean_nu_ext_ie
    p = s_params.p
    bgrE = s_params.bgrE
    bgrI = s_params.bgrI
    externalNoise = s_params.extCurrent_poisson

    # load mft parameters
    nu_vec = m_params.nu_vec
    
    # checks    
    if p < 2:
        sys.exit('ERROR: number of clusters must be >=2. Set Jplus=1 if you want no cluster limit')

    # total number of dynamical populations
    n_dPops_e = MFT_clusteredEINetworks_tools.fcn_n_dynPops(p)
    n_dPops_i = MFT_clusteredEINetworks_tools.fcn_n_dynPops(p)
    n_dPops = n_dPops_e + n_dPops_i

    # vectorize all parameters
    tau_r_vec = tau_r*np.ones(n_dPops)
    tau_m_vec = np.append(tau_m_e*np.ones(n_dPops_e), tau_m_i*np.ones(n_dPops_i))
    tau_s_vec = np.append(tau_s_e*np.ones(n_dPops_e), tau_s_i*np.ones(n_dPops_i))
    Vr_vec = np.append(Vr_e*np.ones(n_dPops_e), Vr_i*np.ones(n_dPops_i))
    Vth_vec = np.append(Vth_e*np.ones(n_dPops_e), Vth_i*np.ones(n_dPops_i))    
    nu_ext = np.append(nu_ext_ee*np.ones(n_dPops_e), nu_ext_ie*np.ones(n_dPops_i))

    # compute weight and degree matrices of each type
    Jmat_rec, Cmat_rec, Jmat_ext, Cmat_ext = MFT_clusteredEINetworks_tools.fcn_compute_final_weight_degree_mats(s_params)    

    # solution
    sol = MFT_basic_tools.fcn_MFT_rate_roots(nu_vec, nu_ext, \
                                             Cmat_rec, Jmat_rec, Jmat_ext, Cmat_ext, externalNoise, \
                                             tau_r_vec, tau_m_vec, tau_s_vec, Vr_vec, Vth_vec) 
           
            
    # check solution
    if sol.success == False:
    
        print('error in root finding!')

        results = {}
        results['nu_out'] = np.ones(n_dPops)*np.nan
        results['Jmat_rec'] = Jmat_rec
        results['Cmat_rec'] = Cmat_rec
        results['Jmat_ext'] = Jmat_ext
        results['Cmat_ext'] = Cmat_ext
        results['m_params'] = m_params
            
        return results
    
    # output the rates
    nu_e_out = sol.x[:n_dPops_e]
    nu_i_out = sol.x[n_dPops_e:]
    nu_out = np.append(nu_e_out, nu_i_out)
    
    # compute self-consistent mu and sigma
    Mu = MFT_basic_tools.fcn_compute_Mu(nu_out, nu_ext, Jmat_rec, Cmat_rec, Jmat_ext, Cmat_ext, tau_m_vec)
    Sigma2 = MFT_basic_tools.fcn_compute_Sigma2(nu_out, nu_ext, Jmat_rec, Cmat_rec, Jmat_ext, Cmat_ext, tau_m_vec, externalNoise)
    
    # check that rate computed using self-consistent mu and sigma matches root solver

    # compute self-consistent rate
    nu_sc = np.zeros(n_dPops)
    for i in range(0,n_dPops):
        nu_sc[i] = MFT_basic_tools.fcn_compute_rate(Vr_vec[i], Vth_vec[i], Mu[i], np.sqrt(Sigma2[i]), \
                                                    tau_r_vec[i], tau_m_vec[i], tau_s_vec[i])  

    # verify that rates are consistent
    nu_check = all(abs(nu_sc - nu_out) < 1e-4)
        
    # check
    if (nu_check == True):
        print('verified solution is self consistent.')
    else:
        sys.exit('ERROR: Solution is not self-consistent!')

    # compute stability
    S, eigenvals_S, realPart_eigvals_S = MFT_basic_tools.fcn_stability_matrix_v1(nu_out, tau_m_vec, tau_s_vec, Vr_vec, Vth_vec, nu_ext, \
                            Jmat_rec, Cmat_rec, Jmat_ext, Cmat_ext, externalNoise)
    
    if nu_check == False:
        S = np.nan*S
        eigenvals_S = np.nan*eigenvals_S
        realPart_eigvals_S = np.nan*realPart_eigvals_S
    
    # get populations that exist
    Epops_exist = MFT_clusteredEINetworks_tools.fcn_find_existing_pops(bgrE, p)
    Ipops_exist = MFT_clusteredEINetworks_tools.fcn_find_existing_pops(bgrI, p)
    Enonexist = np.nonzero(Epops_exist==0)[0]
    Inonexist = n_dPops_e + np.nonzero(Ipops_exist==0)[0]
    nu_out[Enonexist] = 0.
    nu_out[Inonexist] = 0.
                        
    # RESULTS DICTIONARY
    results = {}
    results['Jmat_rec'] = Jmat_rec
    results['Cmat_rec'] = Cmat_rec
    results['Jmat_ext'] = Jmat_ext
    results['Cmat_ext'] = Cmat_ext
    results['nu_out'] = nu_out
    results['Mu'] = Mu
    results['Sigma2'] = Sigma2
    results['m_params'] = m_params
    results['sol'] = sol
    results['S'] = S
    results['realPart_eigvals_S'] = realPart_eigvals_S
                        
    return results


#%% MAIN FUNCTION TO COMPUTE MFT SOLUTION (USING DYNAMICAL EQUATIONS)

def solveMFT_fixedInDeg_EI_net_dynEqs(s_params, m_params):    
    
    # load simulation parameters       
    tau_r = s_params.tau_r              # refractory period
    tau_m_e = s_params.tau_m_e          # membrane time constant E
    tau_m_i = s_params.tau_m_i          # membrane time constant I
    tau_s_e = s_params.tau_s_e          # synaptic time constant E
    tau_s_i = s_params.tau_s_i          # synaptic time constant I
    Vr_e = s_params.Vr_e                # reset potential E
    Vr_i = s_params.Vr_i                # reset potential E
    Vth_e = s_params.Vth_e              # threshold potential E
    Vth_i = s_params.Vth_i              # threshold potential I
    nu_ext_ee = s_params.mean_nu_ext_ee
    nu_ext_ie = s_params.mean_nu_ext_ie
    p = s_params.p
    bgrE = s_params.bgrE
    bgrI = s_params.bgrI
    externalNoise = s_params.extCurrent_poisson

    # load mft parameters
    nu_vec = m_params.nu_vec
    nSteps = m_params.nSteps_MFT_DynEqs
    dt = m_params.dt_MFT_DynEqs
    Te = m_params.tau_e_MFT_DynEqs
    Ti = m_params.tau_i_MFT_DynEqs
    stop_thresh = m_params.stopThresh_MFT_DynEqs
    plot = m_params.plot_MFT_DynEqs    

    # checks    
    if p < 2:
        sys.exit('ERROR: number of clusters must be >=2. Set Jplus=1 if you want no cluster limit')

    # total number of dynamical populations
    n_dPops_e = MFT_clusteredEINetworks_tools.fcn_n_dynPops(p)
    n_dPops_i = MFT_clusteredEINetworks_tools.fcn_n_dynPops(p)
    n_dPops = n_dPops_e + n_dPops_i

    # vectorize all parameters
    tau_r_vec = tau_r*np.ones(n_dPops)
    tau_m_vec = np.append(tau_m_e*np.ones(n_dPops_e), tau_m_i*np.ones(n_dPops_i))
    tau_s_vec = np.append(tau_s_e*np.ones(n_dPops_e), tau_s_i*np.ones(n_dPops_i))
    Vr_vec = np.append(Vr_e*np.ones(n_dPops_e), Vr_i*np.ones(n_dPops_i))
    Vth_vec = np.append(Vth_e*np.ones(n_dPops_e), Vth_i*np.ones(n_dPops_i))    
    nu_ext = np.append(nu_ext_ee*np.ones(n_dPops_e), nu_ext_ie*np.ones(n_dPops_i))

    # MFT timescale
    if ( ( np.size(Te) == 1 ) ):
        Te = Te*np.ones(n_dPops_e)
    if ( ( np.size(Ti) == 1 ) ):
        Ti = Ti*np.ones(n_dPops_i)
    if ( ( np.size(Te)>n_dPops ) ):
        Te = Te[0]*np.ones(n_dPops_e)
    if ( ( np.size(Ti)>n_dPops ) ):
        Ti = Ti[0]*np.ones(n_dPops_i)
        
    T = np.append(Te, Ti)

    # compute weight and degree matrices of each type
    Jmat_rec, Cmat_rec, Jmat_ext, Cmat_ext = MFT_clusteredEINetworks_tools.fcn_compute_final_weight_degree_mats(s_params)  
            
    
    # solution
    nu_out = MFT_basic_tools.fcn_compute_MFT_rates_DynEqs(nSteps, dt, T, stop_thresh, plot, \
                                                                            tau_r_vec, tau_m_vec, tau_s_vec, Vr_vec,  Vth_vec,  \
                                                                                Jmat_rec, Cmat_rec, Jmat_ext, Cmat_ext, externalNoise, nu_ext, nu_vec)              
    
        

    # compute self-consistent mu and sigma
    Mu = MFT_basic_tools.fcn_compute_Mu(nu_out, nu_ext, Jmat_rec, Cmat_rec, Jmat_ext, Cmat_ext, tau_m_vec)
    Sigma2 = MFT_basic_tools.fcn_compute_Sigma2(nu_out, nu_ext, Jmat_rec, Cmat_rec, Jmat_ext, Cmat_ext, tau_m_vec, externalNoise)

    # self-consistent rates
    nu_sc = np.zeros(n_dPops)
    
    for i in range(0,n_dPops):
        
        nu_sc[i] = MFT_basic_tools.fcn_compute_rate(Vr_vec[i], Vth_vec[i], Mu[i], np.sqrt(Sigma2[i]), \
                                                    tau_r_vec[i], tau_m_vec[i], tau_s_vec[i])  
    
    # verify that rates are consistent
    nu_check = all(abs(nu_sc - nu_out) < 1e-4)
        
    if (nu_check == True):
        print('verified solution is self consistent.')
        
    else:
        print('ERROR: Solution is not self-consistent!')
        nu_out = np.nan*nu_out

    # compute stability
    S, eigenvals_S, realPart_eigvals_S = MFT_basic_tools.fcn_stability_matrix_v1(nu_out, tau_m_vec, tau_s_vec, Vr_vec, Vth_vec, nu_ext, \
                            Jmat_rec, Cmat_rec, Jmat_ext, Cmat_ext, externalNoise)
    
    if nu_check == False:
        S = np.nan*S
        eigenvals_S = np.nan*eigenvals_S
        realPart_eigvals_S = np.nan*realPart_eigvals_S

    # get populations that exist
    Epops_exist = MFT_clusteredEINetworks_tools.fcn_find_existing_pops(bgrE, p)
    Ipops_exist = MFT_clusteredEINetworks_tools.fcn_find_existing_pops(bgrI, p)
    Enonexist = np.nonzero(Epops_exist==0)[0]
    Inonexist = n_dPops_e + np.nonzero(Ipops_exist==0)[0]
    nu_out[Enonexist] = 0.
    nu_out[Inonexist] = 0.

    # RESULTS DICTIONARY
    results = {}
    results['Jmat_rec'] = Jmat_rec
    results['Cmat_rec'] = Cmat_rec
    results['Jmat_ext'] = Jmat_ext
    results['Cmat_ext'] = Cmat_ext
    results['nu_out'] = nu_out
    results['Mu'] = Mu
    results['Sigma2'] = Sigma2
    results['m_params'] = m_params
    results['S'] = S
    results['realPart_eigvals_S'] = realPart_eigvals_S
                        
    return results
