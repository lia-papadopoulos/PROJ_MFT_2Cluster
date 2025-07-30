from types import SimpleNamespace
import h5py

def fcn_load_simulations(file_name):

    sim_params = SimpleNamespace()

    with h5py.File(file_name, 'r') as f:

        spikes = f['spikes'][:]

        sim_params_group = f['simulation_parameters']

        for attr_name, attr_value in sim_params_group.attrs.items():
            setattr(sim_params, attr_name, attr_value)

    return sim_params, spikes