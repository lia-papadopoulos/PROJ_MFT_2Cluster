

import numpy as np
import sys
import src.simulation_tools.fcn_stimulation as fcn_stimulation

#%%

'''
assumes that external stimulus enters voltage equation, not poisson

should yield results consistent with BRIAN2, as long as 
refractory time in BRIAN2 is 1 time step longer than refractory time here.

equations take the form of Brunel & Sergi 1998

'''

def fcn_simulate_expSyn(sim_params, J):

#------------------------------------------------------------------------------
# GET PARAMETERS
#------------------------------------------------------------------------------

    save_voltage = sim_params.save_voltage # whether or not to save voltage array

    T0 = sim_params.T0                  # simulation start time
    TF = sim_params.TF                  # simulation end time
    dt = sim_params.dt                  # time step
    N = sim_params.N                    # total number of neurons
    Ne = sim_params.N_e                 # number excitatory
    Ni = sim_params.N_i                 # number inhibitory
    Vth_e = sim_params.Vth_e            # threshold potential E
    Vth_i = sim_params.Vth_i            # threshold potential I
    Vr_e = sim_params.Vr_e              # reset potential E
    Vr_i = sim_params.Vr_i              # reset potential E
    iV = sim_params.iV
    tau_r = sim_params.tau_r            # refractory period
    tau_m_e = sim_params.tau_m_e        # membrane time constant E
    tau_m_i = sim_params.tau_m_i        # membrane time constant I
    tau_s_e = sim_params.tau_s_e        # synaptic time constant E
    tau_s_i = sim_params.tau_s_i        # synaptic time constant I
    t_delay = sim_params.t_delay        # synaptic delay
    pext_ee = sim_params.pext_ee              # external connection probability
    pext_ie = sim_params.pext_ie              # external connection probability
    pext_ei = sim_params.pext_ei              # external connection probability
    pext_ii = sim_params.pext_ii              # external connection probability
    Jee_ext = sim_params.Jee_ext        # external E to E weight
    Jie_ext = sim_params.Jie_ext        # external E to I weight
    Jei_ext = sim_params.Jei_ext
    Jii_ext = sim_params.Jii_ext
    nu_ext_ee = sim_params.nu_ext_ee      # avg baseline afferent rate to E & I neurons [spk/s]
    nu_ext_ie = sim_params.nu_ext_ie      
    nu_ext_ei = sim_params.nu_ext_ei  
    nu_ext_ii = sim_params.nu_ext_ii  
    base_extCurrent_poisson = sim_params.extCurrent_poisson
    # stimulation parameters
    stim_shape = sim_params.stim_shape
    stim_onset = sim_params.stim_onset
    stim_Ecells = sim_params.stim_Ecells
    stim_Icells = sim_params.stim_Icells
    
#------------------------------------------------------------------------------
# SETUP
#------------------------------------------------------------------------------
     
    # THRESHOLD VOLTAGE VECTORS   
    Vth_e_vec = Vth_e*np.ones(Ne)
    Vth_i_vec = Vth_i*np.ones(Ni)
    Vth = np.concatenate((Vth_e_vec, Vth_i_vec))
    # RESET VOLTAGE VECTOR
    Vr_e_vec = Vr_e*np.ones(Ne)
    Vr_i_vec = Vr_i*np.ones(Ni)
    Vr = np.concatenate((Vr_e_vec, Vr_i_vec))
    # MEMBRANE TIME CONSTANT VECTOR
    tau_m_e_vec = tau_m_e*np.ones(Ne)
    tau_m_i_vec = tau_m_i*np.ones(Ni)
    tau_m = np.concatenate((tau_m_e_vec, tau_m_i_vec))    
    # SYNAPTIC WEIGHTS  
    Jij = J.copy()
    
    
        
#------------------------------------------------------------------------------
# INITIAL CONDITIONS
#------------------------------------------------------------------------------ 


    # time each neuron has left to be refractory
    time_ref = np.zeros(N)

#------------------------------------------------------------------------------
# SIMULATION STUFF
#------------------------------------------------------------------------------  

    timePts = np.arange(T0,TF+dt,dt)  # time points of simulation
    nSteps = np.size(timePts)-1       # number of time steps is one less than number of time points in simulation
    spikes = np.zeros((2,1))*np.nan   # initialize array for spike times
        
#------------------------------------------------------------------------------
# SET UP PROPAGATOR FOR SUBTHRESHOLD DYNAMICS
#------------------------------------------------------------------------------ 

    propagator = np.zeros((N, 4, 4))
    
    propagator[:,0,0] = np.exp(-dt/tau_m)
    propagator[:,0,1] = (tau_s_e*tau_m)/(tau_s_e - tau_m) * ( np.exp(-dt/tau_s_e) - np.exp(-dt/tau_m) )
    propagator[:,0,2] = (tau_s_i*tau_m)/(tau_s_i - tau_m) * ( np.exp(-dt/tau_s_i) - np.exp(-dt/tau_m) )
    propagator[:,0,3] = tau_m*(1 - np.exp(-dt/tau_m))
    
    propagator[:,1,0] = 0
    propagator[:,1,1] = np.exp(-dt/tau_s_e)
    propagator[:,1,2] = 0
    propagator[:,1,3] = 0
    
    propagator[:,2,0] = 0
    propagator[:,2,1] = 0
    propagator[:,2,2] = np.exp(-dt/tau_s_i)
    propagator[:,2,3] = 0
    
    propagator[:,3,0] = 0
    propagator[:,3,1] = 0
    propagator[:,3,2] = 0
    propagator[:,3,3] = 1
    
        
#------------------------------------------------------------------------------
# EXTERNAL CURRENTS [POISSON WITH RATE C_ext*nu_ext]
#------------------------------------------------------------------------------

    if ( (base_extCurrent_poisson == True) ):
        
        rng = np.random.default_rng()
        
        # poisson
        extPoisson_ee_vec = np.zeros((Ne, nSteps+1))
        extPoisson_ie_vec = np.zeros((Ni, nSteps+1))
        extPoisson_ei_vec = np.zeros((Ne, nSteps+1))
        extPoisson_ii_vec = np.zeros((Ni, nSteps+1))
        
        # no deterministic inputs
        extDetermin_vec = np.zeros((N, nSteps+1))
        
        
        for i in range(0, Ne):
           
           totalRate_ee = nu_ext_ee[i]
           extPoisson_ee_vec[i,:] = rng.poisson(totalRate_ee*pext_ee*Ne*dt, nSteps+1)
           
           totalRate_ei = nu_ext_ei[i]
           extPoisson_ei_vec[i,:] = rng.poisson(totalRate_ei*pext_ei*Ni*dt, nSteps+1)

        for i in range(0, Ni):
            
           totalRate_ie = nu_ext_ie[i]
           extPoisson_ie_vec[i,:] = rng.poisson(totalRate_ie*pext_ie*Ne*dt, nSteps+1)
           
           totalRate_ii = nu_ext_ii[i]         
           extPoisson_ii_vec[i,:] = rng.poisson(totalRate_ii*pext_ii*Ni*dt, nSteps+1)
        
        # poisson spike trains for all cells
        extPoisson_vec_xe = np.vstack((extPoisson_ee_vec, extPoisson_ie_vec))
        extPoisson_vec_xi = np.vstack((extPoisson_ei_vec, extPoisson_ii_vec))
        
        
    elif ( (base_extCurrent_poisson == False)):
        
        # no external poisson
        extPoisson_vec_xe = np.zeros((N, nSteps+1))
        extPoisson_vec_xi = np.zeros((N, nSteps+1))
        
        # deterministic inputs
        extDetermin_toE_vec = np.zeros((Ne, nSteps+1))
        extDetermin_toI_vec = np.zeros((Ni, nSteps+1))

        for i in range(0, Ne):
           
           totalRate_ee = nu_ext_ee[i] 
           totalRate_ei = nu_ext_ei[i]

           extDetermin_toE_vec[i,:] = totalRate_ee*pext_ee*Ne*Jee_ext + totalRate_ei*pext_ei*Ni*Jei_ext
           

        for i in range(0, Ni):
            
           totalRate_ie = nu_ext_ie[i]
           totalRate_ii = nu_ext_ii[i]         

           extDetermin_toI_vec[i,:] = totalRate_ie*pext_ie*Ne*Jie_ext + totalRate_ii*pext_ii*Ni*Jii_ext
           
        # stack inputs to E and I
        extDetermin_vec = np.vstack((extDetermin_toE_vec, extDetermin_toI_vec))
               
#------------------------------------------------------------------------------
# STIMULATION SETUP [THIS COULD GO INTO A FUNCTION WITH SIM_PARAMS AS INPUT]
#------------------------------------------------------------------------------
    
    # stimulated cells
    stim_cells = np.concatenate((stim_Ecells, stim_Icells))
    
    # initialize
    Istim = np.zeros((N, nSteps+1))
    
    print(np.shape(Istim))

    if ( (stim_shape == 'box') ):
        
        stim_duration = sim_params.stim_duration
        stimRate_e = sim_params.stimRate_E
        stimRate_i = sim_params.stimRate_I
        stim_amp_e = stimRate_e*(Ne*pext_ee*Jee_ext)*np.ones((Ne))
        stim_amp_i = stimRate_i*(Ne*pext_ie*Jie_ext)*np.ones((Ni))
        stim_amplitude = np.concatenate((stim_amp_e, stim_amp_i))
        
        for tInd in range(0,nSteps+1,1):
        
            t = timePts[tInd]
            Istim_at_t = fcn_stimulation.fcn_box_stimulus(stim_onset, stim_duration, stim_amplitude, t)
            Istim[:,tInd] = Istim_at_t*stim_cells
            
    elif ( (stim_shape == 'linear') ):
        
        stim_duration = sim_params.stim_duration
        stimRate_e = sim_params.stimRate_E
        stimRate_i = sim_params.stimRate_I
        stim_amp_e = stimRate_e*(Ne*pext_ee*Jee_ext)*np.ones((Ne))
        stim_amp_i = stimRate_i*(Ne*pext_ie*Jie_ext)*np.ones((Ni))
        stim_amplitude = np.concatenate((stim_amp_e, stim_amp_i))
        
        for tInd in range(0,nSteps+1,1):
        
            t = timePts[tInd]
            Istim_at_t = fcn_stimulation.fcn_linear_stimulus(stim_onset, stim_duration, stim_amplitude, t)
            Istim[:,tInd] = Istim_at_t*stim_cells
        
    elif stim_shape == 'diff2exp':
        
        taur = sim_params.stim_taur
        taud = sim_params.stim_taud
        stimRate_e = sim_params.stimRate_E
        stimRate_i = sim_params.stimRate_I
        stim_amp_e = stimRate_e*(Ne*pext_ee*Jee_ext)*np.ones((Ne))
        stim_amp_i = stimRate_i*(Ne*pext_ie*Jie_ext)*np.ones((Ni))
        stim_amplitude = np.concatenate((stim_amp_e, stim_amp_i))
        
        for tInd in range(0,nSteps+1,1):
        
            t = timePts[tInd]
            Istim_at_t = fcn_stimulation.fcn_diff2exp_stimulus(stim_onset, stim_amplitude, taur, taud, t)
            Istim[:,tInd] = Istim_at_t*stim_cells
        
    else:
        sys.exit('unspecified stimulus type')
        
    
        
#------------------------------------------------------------------------------
# STATE VARIABLES
#------------------------------------------------------------------------------ 

    if save_voltage == True:
    
        V = np.zeros((N, nSteps+1))
        I_exc = np.zeros((N, nSteps+1))
        I_inh = np.zeros((N, nSteps+1))
    
        # initial conditions
        V[:, 0] = iV
        I_exc[:, 0] = 0.
        I_inh[:, 0] = 0.
        
        # external current for straight to voltage equation
        if base_extCurrent_poisson == True:
            I_o = Istim.copy()
        else:
            I_o = Istim + extDetermin_vec
                
    else:
        
        V = iV
        I_exc = np.zeros(N)
        I_inh = np.zeros(N)

        # external current for straight to voltage equation
        if base_extCurrent_poisson == True:
            I_o = Istim.copy()
        else:
            I_o = Istim + extDetermin_vec

                
#------------------------------------------------------------------------------
# PRINT
#------------------------------------------------------------------------------ 
    print('stim amp = %0.3f' % stim_amplitude[0])
    print('Vr_e = %0.3f' % Vr_e_vec[0])
    print('Vr_i = %0.3f' % Vr_i_vec[0])
    print('Vth_e = %0.3f' % Vth_e_vec[0])
    print('Vth_i = %0.3f' % Vth_i_vec[0])
    print('tau_m_e = %0.5f s' % tau_m_e_vec[0])
    print('tau_m_i = %0.5f s' % tau_m_i_vec[0])
    print('tau_s_e = %0.5f s' % tau_s_e)
    print('tau_s_i = %0.5f s' % tau_s_i)
    print('tau_r = %0.5f s' % tau_r)
        
#------------------------------------------------------------------------------
# INTEGRATE
#------------------------------------------------------------------------------     

    
    # time loop
    for tInd in range(0,nSteps,1):
        
                
        #------------------ SUBTHRESHOLD STATE UPDATE ------------------------#
        
        if save_voltage == True:

            #------------------ STATE UPDATE ---------------------------------#

            #------------------ REFRACTORY CONDITION -----------------------------#
            nonrefrac_id = np.nonzero(np.round(time_ref,6) == 0)[0]
            refrac_id = np.setdiff1d(np.arange(0,N), nonrefrac_id)

            # update voltage of non-refractory neurons
            if np.size(nonrefrac_id) > 0:

                V[nonrefrac_id, tInd+1] =   propagator[nonrefrac_id, 0, 0] * V[nonrefrac_id, tInd] + \
                                            propagator[nonrefrac_id, 0, 1] * I_exc[nonrefrac_id, tInd] + \
                                            propagator[nonrefrac_id, 0, 2] * I_inh[nonrefrac_id, tInd] + \
                                            propagator[nonrefrac_id, 0, 3] * I_o[nonrefrac_id, tInd]
            
            if np.size(refrac_id) > 0:

                V[refrac_id, tInd + 1] = Vr[refrac_id]
            
            
            # update excitatory synaptic current
            I_exc[:, tInd+1] = propagator[:, 1, 1] * I_exc[:, tInd]
            
            # update inhibitory synaptic current
            I_inh[:, tInd+1] = propagator[:, 2, 2] * I_inh[:, tInd]


            #------------------ THRESHOLD -------------------------------------#
            #------------------ GET CELLS THAT FIRED BETWEEN TIND AND TIND+1 --#

            fired_ind = np.nonzero(V[:, tInd+1]>=Vth)[0]

            
            #------------------ STORE THESE NEW SPIKES ------------------------#      
            
            # if spike occurred between tInd and tInd +1, set spike time to tInd
            new_spikes = np.vstack((timePts[tInd]*np.ones(np.size(fired_ind)), fired_ind))
            
            # store spikes (row 1 = times, row 2 = neuron index)
            spikes = np.append(spikes, new_spikes, 1)


            #------------------ UPDATE SYNAPTIC WEIGHTS --> JUMP ------------------#
            
            # get delayed spikes --> update curent at tInd+1
            
            indDelay = tInd - round(t_delay/dt)
            
            if indDelay < 0:
                spikes_for_input = np.zeros(N)
            else:            
                tDelay = timePts[indDelay]
                spikes_for_input = np.zeros(N)
                fired_delay_ind = np.nonzero(spikes[0,:] == tDelay)[0]
                fired_delay = spikes[1,fired_delay_ind]
                spikes_for_input[fired_delay.astype(int)] = 1

            # recurrent excitatory spikes
            dI_exc_rec = np.matmul(Jij[:,:Ne],spikes_for_input[:Ne]*1)/tau_s_e

            # recurrent inhibitory spikes
            dI_inh_rec = np.matmul(Jij[:,Ne:],spikes_for_input[Ne:]*1)/tau_s_i

            # external excitatory spikes 
            dI_ext_xe = np.zeros(N)
            dI_ext_xi = np.zeros(N)

            if indDelay >= 0:
                dI_ext_xe[:Ne] = Jee_ext*extPoisson_vec_xe[:Ne, indDelay]/tau_s_e
                dI_ext_xe[Ne:] = Jie_ext*extPoisson_vec_xe[Ne:, indDelay]/tau_s_e
                dI_ext_xi[:Ne] = Jei_ext*extPoisson_vec_xi[:Ne, indDelay]/tau_s_i
                dI_ext_xi[Ne:] = Jii_ext*extPoisson_vec_xi[Ne:, indDelay]/tau_s_i

            # propagate spikes
            

            # total excitatory input to each cell
            I_exc[:, tInd + 1] = I_exc[:, tInd + 1] + dI_ext_xe + dI_exc_rec
            
            # total inhibitory input to each cell
            I_inh[:, tInd + 1] = I_inh[:, tInd + 1] + dI_ext_xi + dI_inh_rec
    
            #------------------ RESET REFRACTORY NEURONS --------------------------#
    
            # reset neurons who fired
            V[fired_ind, tInd + 1] = Vr[fired_ind]        
            
            # set their time remaining refractory to tau_r
            time_ref[fired_ind] = tau_r

            # update time remaining refractory at each step
            time_ref[refrac_id] = time_ref[refrac_id] - dt
            
            
        else:
            

            #------------------ STATE UPDATE ---------------------------------#

            #------------------ REFRACTORY CONDITION -----------------------------#
            nonrefrac_id = np.nonzero(np.round(time_ref,6) == 0)[0]
            refrac_id = np.setdiff1d(np.arange(0,N), nonrefrac_id)

            # update voltage of non-refractory neurons
            # update the voltage of non-refractory
            if np.size(nonrefrac_id) > 0:
                
                V[nonrefrac_id] = propagator[nonrefrac_id, 0, 0] * V[nonrefrac_id] + \
                                  propagator[nonrefrac_id, 0, 1] * I_exc[nonrefrac_id] + \
                                  propagator[nonrefrac_id, 0, 2] * I_inh[nonrefrac_id] + \
                                  propagator[nonrefrac_id, 0, 3] * I_o[nonrefrac_id, tInd]
                               
                                        
            # update voltage of refractory
            if np.size(refrac_id) > 0:
                
                V[refrac_id] = Vr[refrac_id]
            
            
            # update excitatory synaptic current
            I_exc = propagator[:, 1, 1] * I_exc
            
            # update inhibitory synaptic current
            I_inh = propagator[:, 2, 2] * I_inh


            #------------------ THRESHOLD -------------------------------------#
            #------------------ GET CELLS THAT FIRED BETWEEN TIND AND TIND+1 --#

            fired_ind = np.nonzero(V>=Vth)[0]

    
            #------------------ STORE THESE NEW SPIKES ------------------------#      
            
            # if spike occurred between tInd and tInd +1, set spike time to tInd
            new_spikes = np.vstack((timePts[tInd]*np.ones(np.size(fired_ind)), fired_ind))
            
            # store spikes (row 1 = times, row 2 = neuron index)
            spikes = np.append(spikes, new_spikes, 1)


            #------------------ UPDATE SYNAPTIC WEIGHTS --> JUMP ------------------#
            
            # get delayed spikes --> update curent at tInd+1
            
            indDelay = tInd - round(t_delay/dt)
            
            if indDelay < 0:
                spikes_for_input = np.zeros(N)
            else:            
                tDelay = timePts[indDelay]
                spikes_for_input = np.zeros(N)
                fired_delay_ind = np.nonzero(spikes[0,:] == tDelay)[0]
                fired_delay = spikes[1,fired_delay_ind]
                spikes_for_input[fired_delay.astype(int)] = 1

            # recurrent excitatory spikes
            dI_exc_rec = np.matmul(Jij[:,:Ne],spikes_for_input[:Ne]*1)/tau_s_e

            # recurrent inhibitory spikes
            dI_inh_rec = np.matmul(Jij[:,Ne:],spikes_for_input[Ne:]*1)/tau_s_i

            # external excitatory spikes 
            # external excitatory spikes 
            dI_ext_xe = np.zeros(N)
            dI_ext_xi = np.zeros(N)

            if indDelay >= 0:
                dI_ext_xe[:Ne] = Jee_ext*extPoisson_vec_xe[:Ne, indDelay]/tau_s_e
                dI_ext_xe[Ne:] = Jie_ext*extPoisson_vec_xe[Ne:, indDelay]/tau_s_e
                dI_ext_xi[:Ne] = Jei_ext*extPoisson_vec_xi[:Ne, indDelay]/tau_s_i
                dI_ext_xi[Ne:] = Jii_ext*extPoisson_vec_xi[Ne:, indDelay]/tau_s_i
        
            # propagate spikes
            
            # total excitatory input to each cell
            I_exc = I_exc + dI_ext_xe + dI_exc_rec
            
            # total inhibitory input to each cell
            I_inh = I_inh + dI_ext_xi + dI_inh_rec
            
            #------------------ RESET REFRACTORY NEURONS --------------------------#
    
            # reset neurons who fired
            V[fired_ind] = Vr[fired_ind]       
            
        
            # set their time remaining refractory to tau_r
            time_ref[fired_ind] = tau_r
        
            # update time remaining refractory at each step
            time_ref[refrac_id] = time_ref[refrac_id] - dt
        
        
    # delete first column used to initialize spikes    
    spikes = np.delete(spikes, 0, 1)


    # output
    if save_voltage == True:

        return timePts, spikes, V, I_exc, I_inh, I_o
    
    else:
        
        return spikes
