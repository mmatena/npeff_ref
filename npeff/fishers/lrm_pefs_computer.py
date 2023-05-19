"""Code to compute and save LRM-PEFs."""
import dataclasses
import os
from typing import Optional, Sequence

import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from npeff.models import npeff_models
from npeff.util import hdf5_util


###############################################################################


def expand_batch_dims(batch):
    if isinstance(batch, tf.Tensor):
        return tf.expand_dims(batch, axis=1)
    else:
        return {k: tf.expand_dims(v, axis=1) for k, v in batch.items()}


def flatten_batch_mpefs(batch_pefs: Sequence[tf.Tensor]) -> tf.Tensor:
    # output.shape = [batch_size, n_classes, n_params]
    return tf.concat([
        tf.reshape(p,
                   tf.concat([tf.shape(p)[:2], [-1]], axis=0))
        for p in batch_pefs
    ], axis=-1)


def _compute_mpef_frobenius_norms(flat_batch_pefs: tf.Tensor) -> tf.Tensor:
    # flat_batch_pefs.shape = [batch_size, n_classes, n_params]
    AtA = tf.einsum('bcj,bkj->bck', flat_batch_pefs, flat_batch_pefs)
    sq_norms = tf.reduce_sum(tf.square(AtA), axis=[-2, -1])
    return tf.sqrt(sq_norms)


###############################################################################


@dataclasses.dataclass
class SparseLrmPefComputer:
    model: tf.keras.Model
    variables: Sequence[tf.Variable]

    n_values_per_example: int

    top_k_classes: Optional[int] = None

    vectorized: bool = True

    def __post_init__(self):
        self.n_labels = self.model.num_labels
        self.tf_n_values_per_example = tf.cast(self.n_values_per_example, tf.int32)

        self.should_sort_logits = self.top_k_classes is not None
        self.effective_n_labels = self.top_k_classes if self.top_k_classes is not None else int(self.model.num_labels)

    ############################################################
    # I think needed to be hashable to work with tf.function
    
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    ############################################################

    @tf.function
    def _fisher_single_example(self, single_example_batch):

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.variables)

            logits = npeff_models.compute_logits(self.model, single_example_batch)

            # The batch dimension must be 1 to call the model, so we remove it
            # here.
            logits = tf.squeeze(logits, axis=0)

            og_logits = tf.identity(logits)

            if self.should_sort_logits:
                logits = tf.sort(logits, axis=-1, direction='DESCENDING')

            log_probs = tf.nn.log_softmax(logits, axis=-1)
            probs = tf.nn.softmax(logits, axis=-1)

            weighted_grads = []
            log_probs = [log_probs[i] for i in range(self.effective_n_labels)]
            with tape.stop_recording():
                for i in range(self.effective_n_labels):
                    log_prob = log_probs[i]
                    grad = tape.gradient(log_prob, self.variables)
                    weighted_grad = [tf.sqrt(probs[i]) * g for g in grad]
                    weighted_grads.append(weighted_grad)

        return [tf.stack(g, axis=0) for g in zip(*weighted_grads)], og_logits

    @tf.function
    def compute_dense_lrm_pefs_and_logits_for_batch(self, batch):
        batch = expand_batch_dims(batch)

        if self.vectorized:
            fishers, logits = tf.vectorized_map(self._fisher_single_example, batch)
        else:
            fishers, logits = tf.map_fn(
                self._fisher_single_example,
                batch,
                fn_output_signature=(len(self.variables) * [tf.float32], tf.float32),
            )

        # Fishers are parallel to variables list.
        # Each item's shape is [batch, n_classes, *var_shape]
        return fishers, logits

    @tf.function
    def sparsify_batch_mpefs(self, flat_batch_pefs: tf.Tensor):
        # flat_batch_pefs.shape = [batch_size, n_classes, n_params]
        # TODO: Try to optimize this step. Also might start running into int32 limit issues.

        batch_size = tf.shape(flat_batch_pefs)[0]
        n_params = tf.shape(flat_batch_pefs)[-1]

        doubly_flat_batch_pefs = tf.reshape(flat_batch_pefs, [batch_size, -1])
        _, flat_indices = tf.math.top_k(tf.abs(doubly_flat_batch_pefs), k=self.tf_n_values_per_example)

        flat_indices = tf.sort(flat_indices, axis=-1, direction="ASCENDING")

        values = tf.gather(doubly_flat_batch_pefs, flat_indices, batch_dims=1)

        col_indices = flat_indices // n_params
        col_offsets = tf.stack([
            tf.reduce_sum(tf.cast(col_indices == i, tf.int32), axis=-1)
            for i in range(self.effective_n_labels)
        ], axis=-1)
        col_offsets = tf.concat(
            [
                tf.zeros_like(col_offsets[:, :1]),
                col_offsets,
            ], axis=-1)
        col_offsets = tf.math.cumsum(col_offsets, axis=-1)

        row_indices = flat_indices % n_params

        return values, col_offsets, row_indices

    @tf.function
    def process_batch(self, batch):
        fishers, logits = self.compute_dense_lrm_pefs_and_logits_for_batch(batch)

        flat_fishers = flatten_batch_mpefs(fishers)

        frobenius_norms = _compute_mpef_frobenius_norms(flat_fishers)
        values, col_offsets, row_indices = self.sparsify_batch_mpefs(flat_fishers)

        return {
            'logits': logits,
            'frobenius_norms': frobenius_norms,
            'values': values,
            'col_offsets': col_offsets,
            'row_indices': row_indices,
        }


###############################################################################


@dataclasses.dataclass
class StreamingLrmPefSaver:
    fisher_computer: SparseLrmPefComputer

    n_examples: int

    use_tqdm: bool = True

    def __post_init__(self):
        self.model = self.fisher_computer.model
        self.variables = self.fisher_computer.variables
        self.n_values_per_example = self.fisher_computer.n_values_per_example

        self.n_classes = int(self.model.num_labels)
        self.effective_n_classes = self.fisher_computer.effective_n_labels
        self.n_parameters = int(tf.reduce_sum([tf.size(v) for v in self.variables]).numpy())

        self.n_examples_processed = None

    def _initialize_file(self, file: h5py.File):
        n_examples = self.n_examples
        self.data_grp = file.create_group('data')

        self.data_grp.attrs['n_classes'] = self.effective_n_classes
        # self.data_grp.attrs['n_classes'] = self.n_classes
        self.data_grp.attrs['n_parameters'] = self.n_parameters

        int_ds = lambda n, s: self.data_grp.create_dataset(n, s, dtype=np.int32)
        flt_ds = lambda n, s: self.data_grp.create_dataset(n, s, dtype=np.float32)

        self.labels_ds = int_ds('labels', [n_examples])
        self.logits_ds = flt_ds('logits', [n_examples, self.n_classes])

        self.values_ds = flt_ds('values', [n_examples, self.n_values_per_example])
        self.col_offsets_ds = int_ds('col_offsets', [n_examples, self.effective_n_classes + 1])
        self.row_indices_ds = int_ds('row_indices', [n_examples, self.n_values_per_example])

        self.pef_frobenius_norms_ds = flt_ds('pef_frobenius_norms', [n_examples])

    def _write(self, h5_ds: h5py.Dataset, data: tf.Tensor):
        i1 = self.n_examples_processed
        i2 = min(self.n_examples_processed + data.shape[0], self.n_examples)
        h5_ds[i1:i2] = data[:i2 - i1].numpy().astype(h5_ds.dtype)

    def _write_batch_results_to_file(self, file, batch_dict, labels):
        d = batch_dict

        self._write(self.labels_ds, labels)
        self._write(self.logits_ds, d['logits'])

        self._write(self.values_ds, d['values'])
        self._write(self.col_offsets_ds, d['col_offsets'])
        self._write(self.row_indices_ds, d['row_indices'])

        self._write(self.pef_frobenius_norms_ds, d['frobenius_norms'])

    def _compute_and_save_pefs(self, file: h5py.File, ds: tf.data.Dataset):
        self.n_examples_processed = 0

        self._initialize_file(file)

        for examples, labels in ds:
            batch_dict = self.fisher_computer.process_batch(examples)
            self._write_batch_results_to_file(file, batch_dict, labels)
            self.n_examples_processed += labels.shape[0]
            if self.n_examples_processed >= self.n_examples:
                break

    def compute_and_save_pefs(self, filepath: str, ds: tf.data.Dataset):
        # The dataset should be batched.
        if self.use_tqdm:
            ds = tqdm(ds)

        with h5py.File(filepath, "w") as file:
            self._compute_and_save_pefs(file, ds)
