"""Converts the H matrix of a D-NPEFF decomposition into a sparse representation.

In practice, we found that H matrix tends to be extremely sparse. Storing it via
a sparse representation greatly speeds up the time it takes to load from disk.
"""
import os

from absl import app
from absl import flags

import numpy as np

from npeff.decomp import decomps

from npeff.util.color_util import cu


FLAGS = flags.FLAGS

flags.DEFINE_string("decomposition_filepath", None, "Filepath of the D-NPEFF decomposition to sparsify.")

flags.DEFINE_string("output_filepath", None, "Filepath of the h5 file to be written containing the sparsified decomposition.")

flags.DEFINE_float("H_threshold", None, "Minimum value of H entry to keep.")


def to_csr(A: np.ndarray, threshold: float):
    n_rows, n_cols = A.shape
    row_infos = [0]
    all_values = []
    all_col_inds = []

    for i in range(n_rows):
        row = A[i]
        mask = row >= threshold

        values = row[mask]
        all_values.append(values)

        col_inds, = np.nonzero(mask)
        all_col_inds.append(col_inds)

        row_infos.append(row_infos[-1] + values.shape[0])

    all_values = np.concatenate(all_values, axis=0)
    row_infos = np.array(row_infos, dtype=np.int64)
    all_col_inds = np.concatenate(all_col_inds, axis=0).astype(np.int32)

    return all_values, row_infos, all_col_inds


def main(_):
    assert FLAGS.H_threshold is not None

    decomp = decomps.DNpeffDecomposition.load(os.path.expanduser(FLAGS.decomposition_filepath))
    decomp.normalize_components_to_unit_norm()

    print(cu.hly(f'Dense size: {decomp.H.shape[0] * decomp.H.shape[1]}'))

    H_vals, H_row_infos, H_col_inds = to_csr(decomp.H, FLAGS.H_threshold)

    print(cu.hly(f'Sparse NNZ: {H_vals.shape[0]}'))

    sp_nmf = decomps.SparseDNpeffDecomposition(
        W=decomp.W,
        H_shape=decomp.H.shape,
        H_values=H_vals,
        H_row_indices=H_row_infos,
        H_column_indices=H_col_inds,
        new_to_old_col_indices=decomp.new_to_old_col_indices,
        n_parameters=decomp.n_parameters,
    )
    sp_nmf.save(os.path.expanduser(FLAGS.output_filepath))


if __name__ == "__main__":
    app.run(main)
