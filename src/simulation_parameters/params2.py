
# basic imports
from types import SimpleNamespace
import numpy as np

# initialize parameters
params = SimpleNamespace()

# for running stimulations
params.save_voltage = False
params.T0 = 0.
params.TF = 1.5
params.dt = 0.5e-4
params.t_delay = 0e-3

# baseline model parameters
params.N_e = 640                      
params.N_i = 160                     
params.Vth_e = 4.86                   
params.Vth_i = 5.98                   
params.Vr_e = 0.                      
params.Vr_i = 0.                      
params.tau_m_e = 20e-3                
params.tau_m_i = 20e-3                  
params.tau_s_e = 5e-3
params.tau_s_i = 5e-3         
params.tau_r = 5e-3                   
params.extCurrent_poisson = True
params.mean_nu_ext_ee = 7.0
params.mean_nu_ext_ie = 7.0
params.mean_nu_ext_ei = 0.0
params.mean_nu_ext_ii = 0.0
params.net_type = 'cluster'
params.pext_ee = 0.2
params.pext_ie = 0.2
params.pext_ei = 0.
params.pext_ii = 0.
params.pee = 0.2
params.pei = 0.5
params.pii = 0.5
params.pie = 0.5
params.jee = 0.8
params.jie = 2.5
params.jei = -10.6
params.jii = -9.7
params.jie_ext = 12.9
params.jee_ext = 14.5
params.jei_ext = 0.
params.jii_ext = 0.
params.p = 2
params.bgrE = 0.75
params.bgrI = 1. 
params.depress_interCluster = False
params.JplusEE = 6.                 
params.JplusII = 1.0                  
params.JplusEI = 1.0                 
params.JplusIE = 1.0                 

# stimulation
params.stim_shape = 'box'
params.stim_onset = 1.
params.stim_duration = 0.
params.stim_rel_amp = 0.
params.f_Ecells_target = 0.   
params.f_Icells_target = 0.                 
params.f_selectiveClus = 0.


# parameter sweep for simulations
params.n_sweepParams = 1
params.sweep_param1_name = 'JplusEE'
params.sweep_param1_values = np.arange(15,22.5,0.5)