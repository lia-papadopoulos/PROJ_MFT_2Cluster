#%% STANDARD IMPORTS
import importlib
import h5py

# FOR SETTING UP SIMULATION PARAMETERS
from src.simulation_setup import setup_baseline_parameters

# IMPORT MFT FUNCTIONS
from src.MFT_tools.MFT_sweep_JeePlus import fcn_JeePlus_sweep_backwards
from src.MFT_tools.MFT_sweep_JeePlus import fcn_JeePlus_sweep_forwards

# FILE WITH USER INPUTS
import userInput

#%% FUNCTION FOR SAVING SIMULATION OUTPUT

def save_data(output_path, sim_params_name, results_backwards_sweep, results_forwards_sweep, sim_params, mft_params):

    # create hdf5 file
    filename = ( ('%s%s_mft_sweep_JplusEE.h5') % (output_path, sim_params_name) )

    with h5py.File(filename, 'w') as hf:

        # backwards sweep
        new_group = hf.create_group('backwards_sweep')
        for key, value in results_backwards_sweep.items():
            new_group.create_dataset(key, data = value)

        # forwards sweep
        new_group = hf.create_group('forwards_sweep')
        for key, value in results_forwards_sweep.items():
            new_group.create_dataset(key, data = value)

        # simulation parameters
        new_group = hf.create_group('simulation_parameters')
        for key, value in vars(sim_params).items():
            new_group.attrs[key] = value
        
        # mft parameters
        new_group = hf.create_group('mft_parameters')
        for key, value in vars(mft_params).items():
            new_group.attrs[key] = value


#%% DEFINE THE MAIN FUNCTION

def main():

    # get user inputs
    output_path = userInput.output_path
    params_path = userInput.params_path
    sim_params_name = userInput.sim_params_name
    mft_params_name = userInput.mft_params_name

    # load sim params
    sim_params_module = ( ('%s.%s') % (params_path, sim_params_name) )
    sim_params = importlib.import_module(sim_params_module).params 

    # load mft params
    mft_params_module = ( ('%s.%s') % (params_path, mft_params_name) )
    mft_params = importlib.import_module(mft_params_module).mft_params 

    # setup simulation parameters
    setup_baseline_parameters(sim_params)

    #----------------------------------------------------------------------------------------------------------------------------#

    # run the mft
    results_backwards = fcn_JeePlus_sweep_backwards(sim_params, mft_params)
    results_forwards = fcn_JeePlus_sweep_forwards(sim_params, mft_params)

    #----------------------------------------------------------------------------------------------------------------------------#

    # save the results
    save_data(output_path, sim_params_name, results_backwards, results_forwards, sim_params, mft_params)


#%% MAIN FUNCTION

if __name__=='__main__':

    main()

# %%
