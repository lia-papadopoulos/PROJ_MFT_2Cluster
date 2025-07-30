import numpy as np


class Dict2Class:
      
    def __init__(self, my_dict):
          
        for key in my_dict:
            setattr(self, key, my_dict[key])
            
#%%

'''
#---------- FUNCTION FOR GENERATING STIMULUS AT EACH POINT OF SIMULATION -----------#
#
# NOTE: ONLY DOES BOX STIMULUS FOR NOW
# NOTE: SETUP TO WORK WITH THE FUNCTION 'fcn_simulation.py' in PROJ_VariabilityGainMod

# inputs
#    stim_type
#    onset_time
#    duration
#    ampltidue:           either a scalar or an Nx1 array indicating stim amplitude for each cell
#    stimulated_cells:    Nx1 array indicating whether each neuron receives stimulus
#    current_time:        current simulation time
#    
#    potentially other parameters involving stimulus

def fcn_generate_stimulus(stim_type, onset_time, duration, amplitude, stimulated_cells, current_time):
    
    # number of cells
    N = np.size(stimulated_cells)
    
    # compute offset time
    offset_time = onset_time + duration
    
    # check if current time is >= stim onset time and <= stim offset time
    if ( (current_time >= onset_time) and (current_time <= offset_time) ):
        
        if stim_type == 'box':
            
            stim_current = amplitude*stimulated_cells
            
        elif stim_type == 'linear':
            
            # equation for a straight line that satisfies:
            # y(offset_time) = amplitude
            # y(onset_time) = 0
            
            # slope
            m = amplitude/duration
            # intercept
            b = -m*onset_time
            
            stim_current = (m*current_time + b)*stimulated_cells
            

        else:
            
            sys.exit('error: this function only supports box stimulus right now.')
            
            
    else:
            
        stim_current = np.zeros(N)
            
        


    # return
    return stim_current

'''

#%% DIFFERENT TYPES OF STIMULI

def fcn_box_stimulus(onset_time, duration, amplitude, current_time):
    
    # compute offset time
    offset_time = onset_time + duration
    
    # determine if stim should be on
    if ((current_time >= onset_time) and (current_time <= offset_time)):
        
        stim_current = amplitude
        
    else:
        
        stim_current = 0.*amplitude
        
    return stim_current


def fcn_linear_stimulus(onset_time, duration, amplitude, current_time):
    
    # compute offset time
    offset_time = onset_time + duration
    
    # determine if stim should be on
    if ((current_time >= onset_time) and (current_time <= offset_time)):
        
        # equation for a straight line that satisfies:
        # y(offset_time) = amplitude
        # y(onset_time) = 0
        
        # slope
        m = amplitude/duration
        # intercept
        b = -m*onset_time
        # current
        stim_current = (m*current_time + b)
    
    else:
        
        stim_current = 0.*amplitude
        
    return stim_current


def fcn_diff2exp_stimulus(onset_time, peak_amplitude, taur, taud, current_time):
    
    # normalization
    alpha = 1/((taur/taud)**(taur/(taud-taur)) - (taur/taud)**(taud/(taud-taur)))
    
    # should stimulus be on?
    if current_time < onset_time:
        
        stim_current = 0.*peak_amplitude
    
    else:
        
        stim_current = alpha*peak_amplitude*( np.exp(-(current_time - onset_time)/taud) - \
                                              np.exp(-(current_time - onset_time)/taur) )
        
    return stim_current


