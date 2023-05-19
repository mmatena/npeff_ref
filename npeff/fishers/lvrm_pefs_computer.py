"""Code to compute and save LVRM-PEFs."""
import dataclasses
import os
from typing import Sequence

import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from npeff.models import npeff_models
from npeff.util import hdf5_util


###############################################################################

def add_batch_dim(example):
    if isinstance(example, tf.Tensor):
        return tf.expand_dims(example, axis=0)
    else:
        return {k: tf.expand_dims(v, axis=0) for k, v in example.items()}


def flatten_example_mpefs(example_pefs: Sequence[tf.Tensor]) -> tf.Tensor:
    # output.shape = [n_classes, n_params]
    return tf.concat([
        tf.reshape(p, [tf.shape(p)[0], -1])
        for p in example_pefs
    ], axis=-1)


def _compute_mpef_frobenius_norm(flat_example_pefs: tf.Tensor) -> tf.Tensor:
    # flat_example_pefs.shape = [n_classes, n_params]
    AtA = tf.einsum('cj,kj->ck', flat_example_pefs, flat_example_pefs)
    sq_norm = tf.reduce_sum(tf.square(AtA))
    return tf.sqrt(sq_norm)

###############################################################################


@dataclasses.dataclass
class SparseLvrmPefComputer:
    model: tf.keras.Model
    variables: Sequence[tf.Variable]

    n_values_per_example: int
    min_prob_class: float
    max_classes: int

    def __post_init__(self):
        self.n_labels = self.model.num_labels
        self.tf_n_values_per_example = tf.cast(self.n_values_per_example, tf.int32)

    ############################################################
    # I think needed to be hashable to work with tf.function
    
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    ############################################################

    @tf.function
    def _compute_dense_lrm_pefs_and_logits_for_example(self, example):
        single_example_batch = add_batch_dim(example)

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.variables)

            logits = npeff_models.compute_logits(self.model, single_example_batch)

            # The batch dimension must be 1 to call the model, so we remove it
            # here.
            logits = tf.squeeze(logits, axis=0)

            log_probs = tf.nn.log_softmax(logits, axis=-1)

            log_probs = tf.sort(log_probs, direction="DESCENDING")
            probs = tf.nn.softmax(log_probs, axis=-1)

            log_probs = log_probs[:self.max_classes]
            probs = probs[:self.max_classes]

            keep_mask = probs >= self.min_prob_class

            kept_log_probs = tf.boolean_mask(log_probs, keep_mask)
            kept_probs = tf.boolean_mask(probs, keep_mask)

            # Delete these variables because I would totally accidentally use
            # them instead of their kept_* versions.
            del log_probs, probs

            kept_rank = tf.reduce_sum(tf.cast(keep_mask, tf.int32))

            fishers_ta = tf.TensorArray(tf.float32, size=kept_rank * len(self.variables), infer_shape=False)

            for i in tf.range(kept_rank):
                log_prob = kept_log_probs[i]
                with tape.stop_recording():
                    grad = tape.gradient(log_prob, self.variables)
                    weighted_grad = [tf.sqrt(kept_probs[i]) * g for g in grad]
                    for j in range(len(self.variables)):
                        fishers_ta = fishers_ta.write(kept_rank * j + i, weighted_grad[j])
                        
        fishers = [
            fishers_ta.gather(kept_rank * j + tf.range(kept_rank))
            for j in range(len(self.variables))
        ]

        # Fishers are parallel to variables list.
        # Each item's shape is [n_classes, *var_shape]
        #
        # The logits do NOT have a batch dimension.
        return fishers, logits

    @tf.function
    def _sparsify_example_mpefs(self, flat_example_pefs: tf.Tensor):
        # flat_batch_pefs.shape = [ n_classes, n_params]
        # TODO: Try to optimize this step. Also might start running into int32 limit issues.
        n_classes = tf.shape(flat_example_pefs)[0]
        n_params = tf.shape(flat_example_pefs)[-1]

        doubly_flat_example_pefs = tf.reshape(flat_example_pefs, [-1])
        _, flat_indices = tf.math.top_k(tf.abs(doubly_flat_example_pefs), k=self.tf_n_values_per_example)

        flat_indices = tf.sort(flat_indices, direction="ASCENDING")

        values = tf.gather(doubly_flat_example_pefs, flat_indices)

        col_indices = flat_indices // n_params
        col_sizes = tf.vectorized_map(
            lambda i: tf.reduce_sum(tf.cast(col_indices == i, tf.int32)),
            tf.range(n_classes))

        row_indices = flat_indices % n_params

        # Returned things do NOT have a batch dimension.
        return values, col_sizes, row_indices

    @tf.function
    def process_example(self, example):
        # The example should NOT have a batch dimension.
        fishers, logits = self._compute_dense_lrm_pefs_and_logits_for_example(example)

        flat_fishers = flatten_example_mpefs(fishers)

        frobenius_norms = _compute_mpef_frobenius_norm(flat_fishers)
        values, col_sizes, row_indices = self._sparsify_example_mpefs(flat_fishers)

        # NOTE: None of the returned values have a batch dimension.
        return {
            'logits': logits,
            'frobenius_norms': frobenius_norms,
            'values': values,
            'col_sizes': col_sizes,
            'row_indices': row_indices,
        }


###############################################################################


@dataclasses.dataclass
class LvrmPefSaver:
    fisher_computer: SparseLvrmPefComputer

    n_examples: int

    use_tqdm: bool = True

    def __post_init__(self):
        self.model = self.fisher_computer.model
        self.variables = self.fisher_computer.variables

        self.n_values_per_example = self.fisher_computer.n_values_per_example
        self.min_prob_class = self.fisher_computer.min_prob_class
        self.n_parameters = int(tf.reduce_sum([tf.size(v) for v in self.variables]).numpy())

        self.n_examples_processed = None
        self.ranks = None
        self.col_sizes = None

    def _initialize_file(self, file: h5py.File):
        n_examples = self.n_examples
        self.data_grp = file.create_group('data')

        self.data_grp.attrs['min_prob_class'] = self.min_prob_class
        self.data_grp.attrs['n_parameters'] = self.n_parameters

        int_ds = lambda n, s: self.data_grp.create_dataset(n, s, dtype=np.int32)
        flt_ds = lambda n, s: self.data_grp.create_dataset(n, s, dtype=np.float32)

        self.labels_ds = int_ds('labels', [n_examples])
        self.logits_ds = flt_ds('logits', [n_examples, self.fisher_computer.n_labels])

        self.values_ds = flt_ds('values', [n_examples, self.n_values_per_example])
        self.row_indices_ds = int_ds('row_indices', [n_examples, self.n_values_per_example])

        self.pef_frobenius_norms_ds = flt_ds('pef_frobenius_norms', [n_examples])

    def _write(self, h5_ds: h5py.Dataset, data: tf.Tensor):
        h5_ds[self.n_examples_processed] = data.numpy().astype(h5_ds.dtype)

    def _store_example_results(self, file, batch_dict, labels):
        # Save some to the file directly, add others to in-RAM cache.
        d = batch_dict

        self.col_sizes.append(d['col_sizes'].numpy())
        self.ranks.append(d['col_sizes'].shape[0])

        self._write(self.labels_ds, labels)
        self._write(self.logits_ds, d['logits'])

        self._write(self.values_ds, d['values'])
        self._write(self.row_indices_ds, d['row_indices'])

        self._write(self.pef_frobenius_norms_ds, d['frobenius_norms'])

    def _finalize_file(self, file: h5py.File):
        col_sizes = np.concatenate(self.col_sizes, axis=0).astype(np.int32)
        col_sizes_ds = self.data_grp.create_dataset('col_sizes', col_sizes.shape, dtype=np.int32)
        col_sizes_ds[:] = col_sizes

        ranks = np.stack(self.ranks, axis=0).astype(np.int32)
        ranks_ds = self.data_grp.create_dataset('ranks', ranks.shape, dtype=np.int32)
        ranks_ds[:] = ranks

    def _compute_and_save_pefs(self, file: h5py.File, ds: tf.data.Dataset):
        self.n_examples_processed = 0
        self.ranks = []
        self.col_sizes = []

        self._initialize_file(file)

        for example, label in ds:
            example_dict = self.fisher_computer.process_example(example)
            self._store_example_results(file, example_dict, label)
            self.n_examples_processed += 1
            if self.n_examples_processed >= self.n_examples:
                break

        self._finalize_file(file)

    def compute_and_save_pefs(self, filepath: str, ds: tf.data.Dataset):
        # The dataset should NOT be batched.
        if self.use_tqdm:
            ds = tqdm(ds)

        with h5py.File(filepath, "w") as file:
            self._compute_and_save_pefs(file, ds)
