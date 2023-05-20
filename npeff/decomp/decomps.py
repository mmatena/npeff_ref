"""Classes holding representations of NPEFF decompositions."""
import dataclasses
import os
from typing import Union

import h5py
import numpy as np

from npeff.util import hdf5_util


###############################################################################
# General-purpose utility functions.


def load_W(nmf_filepath: str):
    with h5py.File(os.path.expanduser(nmf_filepath), "r") as f:
        return hdf5_util.load_h5_ds(f['data/W'])


###############################################################################


@dataclasses.dataclass
class DNpeffDecomposition:
    """Results of a D-NEFF decomposition."""

    # shape = [n_examples, n_components]
    W: np.ndarray
    # shape = [n_components, n_features]
    H: np.ndarray

    # Indices of parameters in original that are kept in the reduced
    # per-example Fishers.
    # shape = [n_features], dtype=np.int32
    new_to_old_col_indices: np.ndarray

    # Equivalent to full dense size.
    n_parameters: int
    
    def save(self, filepath: str):
        with h5py.File(os.path.expanduser(filepath), "w") as f:
            data = f.create_group('data')

            hdf5_util.save_h5_ds(data, 'W', self.W)
            hdf5_util.save_h5_ds(data, 'H', self.H)
            hdf5_util.save_h5_ds(data, 'reduce_kept_indices', self.new_to_old_col_indices)
            data['reduce_kept_indices'].attrs['full_dense_size'] = self.n_parameters

    @classmethod
    def load(cls, filepath: str) -> 'DNpeffDecomposition':
        with h5py.File(os.path.expanduser(filepath), "r") as f:
            ret = cls(
                W=hdf5_util.load_h5_ds(f['data/W']),
                H=hdf5_util.load_h5_ds(f['data/H']),
                new_to_old_col_indices=hdf5_util.load_h5_ds(f['data/reduce_kept_indices']),
                n_parameters=f['data/reduce_kept_indices'].attrs['full_dense_size']
            )

        # NOTE: This is needed due to a bug I had in an early version of the D-NPEFF code.
        # I'm not sure if it is still present.
        if isinstance(ret.n_parameters, np.ndarray):
            assert len(ret.n_parameters) == 1
            ret.n_parameters = ret.n_parameters[0]

        return ret


@dataclasses.dataclass
class SparseDNpeffDecomposition:
    """Results of a D-NEFF decomposition with a sparse representation of the H matrix.
    
    The H-matrix is stored is CSR-format.
    """

    # shape = [n_examples, n_components]
    W: np.ndarray

    # shape = [n_components, n_features], dtype=np.int32
    H_shape: np.ndarray
    # shape = [nnz], dtype=float32
    H_values: np.ndarray
    # shape = [n_examples + 1], dtype=int64
    H_row_indices: np.ndarray
    # shape = [nnz], dtype=int32
    H_column_indices: np.ndarray
    
    # Indices of parameters in original that are kept in the reduced
    # per-example Fishers.
    # shape = [n_features], dtype=np.int32
    new_to_old_col_indices: np.ndarray

    # Equivalent to full dense size.
    n_parameters: int

    def save(self, filepath: str):
        with h5py.File(os.path.expanduser(filepath), "w") as f:
            data = f.create_group('data')

            hdf5_util.save_h5_ds(data, 'W', self.W)

            hdf5_util.save_h5_ds(data, 'reduce_kept_indices', self.new_to_old_col_indices)
            data['reduce_kept_indices'].attrs['full_dense_size'] = self.n_parameters

            H_group = data.create_group('H')
            H_group.attrs['shape'] = self.H_shape

            hdf5_util.save_h5_ds(H_group, 'values', self.H_values)
            hdf5_util.save_h5_ds(H_group, 'row_indices', self.H_row_indices)
            hdf5_util.save_h5_ds(H_group, 'column_indices', self.H_column_indices)

    @classmethod
    def load(cls, filepath: str) -> 'SparseDNpeffDecomposition':
        with h5py.File(os.path.expanduser(filepath), "r") as f:
            return cls(
                W=hdf5_util.load_h5_ds(f['data/W']),
                H_shape=f['data/H'].attrs['shape'],
                H_values=hdf5_util.load_h5_ds(f['data/H/values']),
                H_row_indices=hdf5_util.load_h5_ds(f['data/H/row_indices']),
                H_column_indices=hdf5_util.load_h5_ds(f['data/H/column_indices']),
                new_to_old_col_indices=hdf5_util.load_h5_ds(f['data/reduce_kept_indices']),
                n_parameters=f['data/reduce_kept_indices'].attrs['full_dense_size']
            )


###############################################################################


@dataclasses.dataclass
class LrmNpeffDecomposition:
    """Results of an L(V)RM-NEFF decomposition"""

    # shape = [n_examples, n_components]
    W: np.ndarray

    # Is None if we don't load it.
    # shape = [n_components, n_features]
    G: Union[np.ndarray, None]

    # Indices of parameters in original that are kept in the reduced
    # per-example Fishers.
    # shape = [n_features], dtype=np.int32
    new_to_old_col_indices: np.ndarray

    # Equivalent to full dense size.
    n_parameters: int

    n_classes: int

    # Loss information
    log_loss_frequency: int
    losses_G_only: np.ndarray
    losses_joint: np.ndarray

    def normalize_components_to_unit_norm(self, eps=1e-12):
        # Normalizes the G such that the rank-1 basis PSD matrices have unit frobenius
        # norm.
        # NOTE: This will modify W, G in place!
        norms = np.sum(self.G**2, axis=-1, keepdims=True)
        self.G /= np.sqrt(norms) + eps
        self.W *= norms.T + eps

    @classmethod
    def load(cls, filepath: str, *, read_G: bool = True):
        with h5py.File(os.path.expanduser(filepath), "r") as f:
            return cls(
                W=hdf5_util.load_h5_ds(f['data/W']),
                G=hdf5_util.load_h5_ds(f['data/G']) if read_G else None,
                new_to_old_col_indices=hdf5_util.load_h5_ds(f['data/new_to_old_col_indices']),
                n_parameters=f['data'].attrs['n_parameters'],
                n_classes=f['data'].attrs['n_classes'],
                log_loss_frequency=f['losses'].attrs['log_loss_frequency'],
                losses_G_only=hdf5_util.load_h5_ds(f['losses/G_only']),
                losses_joint=hdf5_util.load_h5_ds(f['losses/joint']),
            )

    @classmethod
    def read_loss_info_into_dict(cls, filepath: str):
        with h5py.File(os.path.expanduser(filepath), "r") as f:
            return {
                'log_loss_frequency': f['losses'].attrs['log_loss_frequency'],
                'losses_G_only': hdf5_util.load_h5_ds(f['losses/G_only']),
                'losses_joint': hdf5_util.load_h5_ds(f['losses/joint']),
            }

    def save(self, filepath: str):
        with h5py.File(os.path.expanduser(filepath), "w") as f:
            data = f.create_group('data')

            data.attrs['n_parameters'] = self.n_parameters
            data.attrs['n_classes'] = self.n_classes

            hdf5_util.save_h5_ds(data, 'W', self.W)
            hdf5_util.save_h5_ds(data, 'G', self.G)
            hdf5_util.save_h5_ds(data, 'new_to_old_col_indices', self.new_to_old_col_indices)

            losses = f.create_group('losses')
            losses.attrs['log_loss_frequency'] = self.log_loss_frequency
        
            hdf5_util.save_h5_ds(losses, 'G_only', self.losses_G_only)
            hdf5_util.save_h5_ds(losses, 'joint', self.losses_joint)
