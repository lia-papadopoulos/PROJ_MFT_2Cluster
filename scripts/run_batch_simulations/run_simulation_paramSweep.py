

#%% BASIC IMPORTS
import argparse
import h5py
import importlib
import os

# FOR SETTING UP SIMULATION
from src.simulation_setup import setup_baseline_parameters, set_initial_voltage, setup_stimulation, fcn_swept_param_name_val_str

# NETWORK GENERATION
from src.make_networks.generate_network import generate_network
from src.make_networks.generate_network import get_network_population_info

# FOR RUNNING SIMULATIONS
from src.simulation_tools.fcn_simulation_EIextInput import fcn_simulate_expSyn


#%% FUNCTION TO ADD ARGPARSE PARAMETERS TO SIMULATION PARAMETERS

def get_argparse_parameters(sim_params, args):

    # add argparse values to sim_params
    for key, value in vars(args).items():
        setattr(sim_params, key, value)

    return None


#%% FUNCTION THAT SETS CURRENT VALUE OF SWEPT PARAMETER 

def set_value_swept_parameter(sim_params, args):

    for i in range(0, sim_params.n_sweepParams):

        key_to_param_name = ( ('sweep_param%d_name') % (i+1) )
        paramName = vars(sim_params)[key_to_param_name]

        key_to_param_values = ( ('sweep_param%d_values') % (i+1) )
        paramValue = vars(sim_params)[key_to_param_values][args.indSweep]

        setattr(sim_params, paramName, paramValue)


#%% FUNCTION TO SET SEEDS FOR MODEL INITIALIZATION

def set_seeds(args):

    network_seed = args.indNetwork                   # not stimulus or trial dependent
    stimClusters_seed = args.indNetwork*args.indStim      # not trial dependendent, but does depend on network & stimulus
    stimNeurons_seed = args.indStim                  # not trial dependent
    initialCondition_seed = args.indTrial            # different for every trial
    
    return network_seed, stimClusters_seed, stimNeurons_seed, initialCondition_seed

#%% FUNCTION FOR SAVING SIMULATION OUTPUT

def save_data(params, args, spikes, network_seed, stimClusters_seed, stimNeurons_seed, initialVoltage_seed):

    # create string of swept parameters and their values for this simulation
    sweep_param_str = fcn_swept_param_name_val_str(params, args.indSweep)

    # make output directory if it doesn't exist
    directory_path = ( ('%s%s/') % (args.output_path, args.sim_params_name) )
    os.makedirs(directory_path, exist_ok=True)

    # create hdf5 file
    filename = ( ('%s%s_sweep_%s_network%d_stim%d_trial%d.h5') % (directory_path, args.sim_params_name, sweep_param_str, args.indNetwork, args.indStim, args.indTrial) )
    with h5py.File(filename, 'w') as hf:

        # spike times
        hf.create_dataset('spikes', data=spikes)
        
        # simulation parameters
        group_simParams = hf.create_group('simulation_parameters')
        for key, value in vars(params).items():
            group_simParams.attrs[key] = value
        
        # save seeds
        group_simParams.attrs['sim_params_name'] = args.sim_params_name
        group_simParams.attrs['network_seed'] = network_seed
        group_simParams.attrs['stimClusters_seed'] = stimClusters_seed
        group_simParams.attrs['stimNeurons_seed'] = stimNeurons_seed
        group_simParams.attrs['initialVoltage_seed'] = initialVoltage_seed



#%% DEFINE MAIN FUNCTION

def main(args):

    #----------------------------------------------------------------------------------------------------------------------------#

    # import simulation parameters
    sim_params_module = ( ('%s.%s') % (args.sim_params_path, args.sim_params_name) )
    params = importlib.import_module(sim_params_module).params 

    # get information from argparse
    get_argparse_parameters(params, args)

    # set value of swept parameter
    set_value_swept_parameter(params, args)

    # seeds for model initialization
    network_seed, stimClusters_seed, stimNeurons_seed, initialVoltage_seed = set_seeds(args)

    # basic setup
    setup_baseline_parameters(params)

    # make network
    W = generate_network(params, network_seed)

    # get network population info
    get_network_population_info(params)

    # set initial voltage
    set_initial_voltage(params, initialVoltage_seed)
        
    # setup stimulus
    setup_stimulation(params, stimClusters_seed, stimNeurons_seed)    

    #----------------------------------------------------------------------------------------------------------------------------#

    # run simulation
    params.save_voltage = False
    spikes = fcn_simulate_expSyn(params, W)

    #----------------------------------------------------------------------------------------------------------------------------#

    # save the data
    save_data(params, args, spikes, network_seed, stimClusters_seed, stimNeurons_seed, initialVoltage_seed)


#%%
# main function
if __name__=='__main__':

    # initialize arg parser
    parser = argparse.ArgumentParser() 

    parser.add_argument('-output_path', '--output_path', type=str)
    parser.add_argument('-sim_params_path', '--sim_params_path', type=str)
    parser.add_argument('-sim_params_name', '--sim_params_name', type=str)
    parser.add_argument('-indNetwork', '--indNetwork', type=int)
    parser.add_argument('-indStim', '--indStim', type=int)
    parser.add_argument('-indTrial', '--indTrial', type=int)
    parser.add_argument('-indSweep', '--indSweep', type=int)

    args = parser.parse_args()

    # run main function
    main(args)
    