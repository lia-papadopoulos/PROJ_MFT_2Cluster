
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
params.N_e = 640*4                      
params.N_i = 160*4                     
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
params.bgrE = 0.
params.bgrI = 0.0
params.depress_interCluster = False
params.JplusEE = 7                
params.JplusII = 7                  
params.JplusEI = 1.                 
params.JplusIE = 7                 

# stimulation
params.stim_shape = 'box'
params.stim_onset = 1.
params.stim_duration = 0.
params.stim_rel_amp = 0.
params.f_Ecells_target = 0.   
params.f_Icells_target = 0.                 
params.f_selectiveClus = 0.


# parameter sweep for simulations
params.n_sweepParams = 3
'''
params.sweep_param_name_dict = {}
params.sweep_param_name_dict['param1'] = 'JplusEE'
params.sweep_param_name_dict['param2'] = 'JplusIE'
params.sweep_param_name_dict['param3'] = 'JplusII'
params.sweep_param_val_dict = {}
params.sweep_param_val_dict['param1'] = np.arange(3,10,1)
params.sweep_param_val_dict['param2'] = np.arange(3,10,1)
params.sweep_param_val_dict['param3'] = np.arange(3,10,1)
'''
params.sweep_param1_name = 'JplusEE'
params.sweep_param2_name = 'JplusIE'
params.sweep_param3_name = 'JplusII'
params.sweep_param1_values = np.arange(3,10,1)
params.sweep_param2_values = np.arange(3,10,1)
params.sweep_param3_values = np.arange(3,10,1)