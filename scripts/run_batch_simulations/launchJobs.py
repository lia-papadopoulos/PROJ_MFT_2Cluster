# LAUNCH JOBS FOR PARAMETER SWEEP

# BASIC IMPORTS
import os
import numpy as np
import importlib

# USER-DEFINED INFO FOR SIMULATION
import userSettings as simInfo

# UNPACK SIM INFO
output_path = simInfo.output_path
sim_params_path = simInfo.sim_params_path
sim_params_name = simInfo.sim_params_name
maxCores = simInfo.maxCores
cores_per_job = simInfo.cores_per_job
indNet_start = simInfo.indNet_start
nNetworks = simInfo.nNetworks
nStim = simInfo.nStim
nTrials = simInfo.nTrials

# LOAD SIMULATION PARAMETERS
sim_params_module = ( ('%s.%s') % (sim_params_path, sim_params_name) )
params = importlib.import_module(sim_params_module).params 

# UNPACK SIMULATION PARAMETERS
sweep_param1_values = params.sweep_param1_values

# SET NUMBER OF SIMULTANEOUS JOBS
simul_jobs = round(maxCores/cores_per_job)

# FUNCTION TO RUN JOBS
def launchJobs():
      
    # tell task-spooler how many jobs it can run simultaneously
    os.system("tsp -S %s" % simul_jobs)
    
    # number of parameter values
    nParam_vals = np.size(sweep_param1_values)

    # loop over swept parameter, networks and launch jobs
    for ind_param in range(0, nParam_vals):

        # loop over networks    
        for indNetwork in range(indNet_start, indNet_start + nNetworks):

            # loop over stimuli
            for indStim in range(0, nStim):
            
                # loop over initial conditions
                for indTrial in range(0,nTrials):

                    # arguments to pass main simulation function
                    str_pass = (' --output_path %s --sim_params_path %s --sim_params_name %s --indNetwork %d --indStim %d --indTrial %d --indSweep %d') 
                    tuple_pass = (output_path, sim_params_path, sim_params_name, indNetwork, indStim, indTrial, ind_param)
                    
                    # run simulations
                    command = 'tsp python run_simulation_paramSweep.py' + ( str_pass % tuple_pass )
                    
                    # SUBMIT JOBS
                    os.system(command) 

    
# CALL JOB LAUNCHER
launchJobs()


