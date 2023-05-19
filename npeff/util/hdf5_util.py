import numpy as np


def set_h5_ds(ds, val):
    # NOTE: Code modified from a section of tf source code here.
    if not val.shape:
        # scalar
        ds[()] = val
    else:
        ds[:] = val


def save_h5_ds(group, name, ndarray):
    ds = group.create_dataset(name, ndarray.shape, dtype=ndarray.dtype)
    set_h5_ds(ds, ndarray)
    return ds


def load_h5_ds(ds):
    array = np.empty(ds.shape, dtype=ds.dtype)
    if array.size > 0:
        ds.read_direct(array)
    return array
