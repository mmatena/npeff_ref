"""Dataset-level Fisher information matrix approximations."""
import os
import dataclasses

import h5py
import numpy as np

from npeff.util import hdf5_util


@dataclasses.dataclass
class DenseDDsf:
    """Diagonal approximation of a dataset-level Fisher matrix."""

    # fisher.shape = [n_parameters]
    fisher: np.ndarray

    def normalize_to_unit_norm(self, eps=1e-12):
        """Normalize the fisher such that it has unit L2 norm."""
        # NOTE: This will modify self.fisher in place!
        self.fisher /= np.sqrt(np.sum(self.fisher**2)) + eps

    def save(self, filepath: str):
        with h5py.File(os.path.expanduser(filepath), "w") as f:
            data = f.create_group('data')
            hdf5_util.save_h5_ds(data, 'fisher', self.fisher)

    @classmethod
    def load(cls, filepath: str) -> 'DenseDDsf':
        with h5py.File(os.path.expanduser(filepath), "r") as f:
            return cls(
                fisher=hdf5_util.load_h5_ds(f['data/fisher']),
            )
