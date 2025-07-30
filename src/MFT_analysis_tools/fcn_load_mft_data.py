
from types import SimpleNamespace
import h5py

def fcn_load_mft_sweep_JeePlus(file_name):

    backwards_sweep = SimpleNamespace()
    forwards_sweep = SimpleNamespace()

    with h5py.File(file_name, 'r') as f:

        grp = f['backwards_sweep']

        for key in grp.keys():
            data = grp[key][()]
            setattr(backwards_sweep, key, data)

        grp = f['forwards_sweep']

        for key in grp.keys():
            data = grp[key][()]
            setattr(forwards_sweep, key, data)

    return backwards_sweep, forwards_sweep