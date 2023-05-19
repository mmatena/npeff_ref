"""Classes holding representations of PEFs."""
import os
import dataclasses
from typing import Sequence

import h5py
import numpy as np

from npeff.util import hdf5_util


###############################################################################
# Utilities for loading logits and labels from a PEFs file. They are
# not technically part of the PEFs but we save them with them for convenience.
# This lets us just load the logits and/or labels from a PEFs file if that is
# only what we want.


def load_logits(pefs_filepath: str) -> np.ndarray:
    with h5py.File(os.path.expanduser(pefs_filepath), "r") as f:
        return hdf5_util.load_h5_ds(f['data/logits'])


def load_labels(pefs_filepath: str) -> np.ndarray:
    with h5py.File(os.path.expanduser(pefs_filepath), "r") as f:
        return hdf5_util.load_h5_ds(f['data/labels'])


###############################################################################


@dataclasses.dataclass
class SparseDiagonalPefs:
    """Represents the sparse diagonal PEFs of a data set."""

    n_parameters: int

    # values.shape = [n_examples, nnz_per_example]
    values: np.ndarray
    # indices.shape = [n_examples, nnz_per_example]
    indices: np.ndarray

    # pef_norms.shape = [n_examples]
    pef_norms: np.ndarray

    # labels.shape = [n_examples]
    labels: np.ndarray
    # logits.shape = [n_examples, n_classes]
    logits: np.ndarray

    def create_for_subset(self, subset_inds: Sequence[int]) -> 'SparseDiagonalPefs':
        """Creates a SparseLrmPefs object corresponding to examples with indices from subset_inds."""
        if not isinstance(subset_inds, np.ndarray):
            subset_inds = np.array(subset_inds, dtype=np.int32)
        return SparseDiagonalPefs(
            n_parameters=self.n_parameters,
            values=self.values[subset_inds],
            indices=self.indices[subset_inds],
            pef_norms=self.pef_norms[subset_inds],
            labels=self.labels[subset_inds],
            logits=self.logits[subset_inds],
        )

    def save(self, filepath: str):
        with h5py.File(os.path.expanduser(filepath), "w") as f:
            hdf5_util.save_h5_ds(f, 'data/dense_fisher_norms', self.pef_norms)

            hdf5_util.save_h5_ds(f, 'data/labels', self.labels)
            hdf5_util.save_h5_ds(f, 'data/predicted_logits', self.logits)

            hdf5_util.save_h5_ds(f, 'data/fisher/values', self.values)
            hdf5_util.save_h5_ds(f, 'data/fisher/indices', self.indices)

            f['data/fisher'].attrs['dense_fisher_size'] = self.n_parameters

    @classmethod
    def load(cls, filepath: str):
        with h5py.File(os.path.expanduser(filepath), "r") as f:
            return cls(
                n_parameters=f['data/fisher'].attrs['dense_fisher_size'],
                labels=hdf5_util.load_h5_ds(f['data/labels']),
                logits=hdf5_util.load_h5_ds(f['data/logits']),
                values=hdf5_util.load_h5_ds(f['data/fisher/values']),
                indices=hdf5_util.load_h5_ds(f['data/fisher/indices']),
                pef_norms=hdf5_util.load_h5_ds(f['data/dense_fisher_norms']),
            )

###############################################################################


@dataclasses.dataclass
class SparseLrmPefs:
    """Represents the sparse LRM-PEFs of a data set.

    The each PEF is stored in CSC format. We stored the matrix A_i for each example
    such that the full PEF is equal to A_iA_i^T. We assume that the rank of each
    PEF is the same across all examples. We also assume that the representation
    for each PEF contains the same number of non-zero values.
    """

    n_classes: int
    n_parameters: int

    # values.shape = [n_examples, nnz_per_example]
    values: np.ndarray
    # col_offsets.shape = [n_examples, n_classes + 1]
    col_offsets: np.ndarray
    # row_indices.shape = [n_examples, nnz_pef_example]
    row_indices: np.ndarray

    # pef_norms.shape = [n_examples]
    pef_norms: np.ndarray
    
    # labels.shape = [n_examples]
    labels: np.ndarray
    # logits.shape = [n_examples, n_classes]
    logits: np.ndarray

    def create_for_subset(self, subset_inds: Sequence[int]) -> 'SparseLrmPefs':
        """Creates a SparseLrmPefs object corresponding to examples with indices from subset_inds."""
        if not isinstance(subset_inds, np.ndarray):
            subset_inds = np.array(subset_inds, dtype=np.int32)
        return SparseLrmPefs(
            n_classes=self.n_classes,
            n_parameters=self.n_parameters,
            labels=self.labels[subset_inds],
            logits=self.logits[subset_inds],
            values=self.values[subset_inds],
            col_offsets=self.col_offsets[subset_inds],
            row_indices=self.row_indices[subset_inds],
            pef_norms=self.pef_norms[subset_inds],
        )

    def save(self, filepath: str):
        with h5py.File(os.path.expanduser(filepath), "w") as f:
            hdf5_util.save_h5_ds(f, 'data/labels', self.labels)
            hdf5_util.save_h5_ds(f, 'data/logits', self.logits)

            hdf5_util.save_h5_ds(f, 'data/values', self.values)
            hdf5_util.save_h5_ds(f, 'data/col_offsets', self.col_offsets)
            hdf5_util.save_h5_ds(f, 'data/row_indices', self.row_indices)

            hdf5_util.save_h5_ds(f, 'data/pef_frobenius_norms', self.pef_norms)

            f['data'].attrs['n_classes'] = self.n_classes
            f['data'].attrs['n_parameters'] = self.n_parameters

    @classmethod
    def load(cls, filepath: str):
        with h5py.File(os.path.expanduser(filepath), "r") as f:
            return cls(
                n_classes=f['data'].attrs['n_classes'],
                n_parameters=f['data'].attrs['n_parameters'],
                labels=hdf5_util.load_h5_ds(f['data/labels']),
                logits=hdf5_util.load_h5_ds(f['data/logits']),
                values=hdf5_util.load_h5_ds(f['data/values']),
                col_offsets=hdf5_util.load_h5_ds(f['data/col_offsets']),
                row_indices=hdf5_util.load_h5_ds(f['data/row_indices']),
                pef_norms=hdf5_util.load_h5_ds(f['data/pef_frobenius_norms']),
            )

###############################################################################


@dataclasses.dataclass
class SparseLvrmPefs:
    """Represents the sparse LVRM-PEFs of a data set.
    
    LVRM stands low, variable-rank matrix. This implementation has a
    a fixed NNZ entries per example, which probably leads to examples
    with high rank PEFs have poor sparse approximations.
    """

    n_parameters: int

    # values.shape = [n_examples, nnz_pef_example]
    values: np.ndarray

    # row_indices.shape = [n_examples, nnz_pef_example]
    row_indices: np.ndarray

    # ranks.shape = [n_examples]
    ranks: np.ndarray
    # col_sizes.shape = [sum(ranks)]
    col_sizes: np.ndarray

    # pef_norms.shape = [n_examples]
    pef_norms: np.ndarray

    # labels.shape = [n_examples]
    labels: np.ndarray
    # logits.shape = [n_examples, n_classes]
    logits: np.ndarray

    def create_for_subset(self, subset_inds: Sequence[int]) -> 'SparseLvrmPefs':
        """Creates a SparseLvrmPefs object corresponding to examples with indices from subset_inds."""
        raise NotImplementedError('TODO')

    def save(self, filepath: str):
        raise NotImplementedError('TODO')

    @classmethod
    def load(cls, filepath: str):
        raise NotImplementedError('TODO')
