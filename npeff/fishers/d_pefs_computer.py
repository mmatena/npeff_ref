"""Code to compute and save D-PEFs."""
import dataclasses
import os
from typing import Optional, Sequence, Union

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


def slice_batch(batch, index):
    if isinstance(batch, tf.Tensor):
        return batch[index]
    else:
        return {k: v[index] for k, v in batch.items()}


def batch_size_from_batch(batch):
    if isinstance(batch, tf.Tensor):
        return tf.shape(batch)[0]
    else:
        return tf.shape(list(batch.values())[0])[0]


def flatten_batch_d_pefs(batch_pefs: Sequence[tf.Tensor]) -> tf.Tensor:
    # output.shape = [batch_size, n_params]
    return tf.concat([
        tf.reshape(p,
                   tf.concat([tf.shape(p)[:1], [-1]], axis=0))
        for p in batch_pefs
    ], axis=-1)


def _compute_d_pef_norms(flat_batch_pefs: tf.Tensor) -> tf.Tensor:
    # flat_batch_pefs.shape = [batch_size, n_params]
    return tf.sqrt(tf.einsum('bj,bj->b', flat_batch_pefs, flat_batch_pefs))


###############################################################################


@dataclasses.dataclass
class AllClassesSparseDPefComputer:
    """Compute D-PEFs using all classes from the predictive distribution."""
    model: tf.keras.Model
    variables: Sequence[tf.Variable]

    n_values_per_example: int
    
    vectorized: bool = True

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
    def _fisher_single_example(self, single_example_batch):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.variables)

            logits = npeff_models.compute_logits(self.model, single_example_batch)

            # The batch dimension must be 1 to call the model, so we remove it
            # here.
            logits = tf.squeeze(logits, axis=0)

            log_probs = tf.nn.log_softmax(logits, axis=-1)
            probs = tf.nn.softmax(logits, axis=-1)

            sq_grads = []
            log_probs = [log_probs[i] for i in range(self.n_labels)]
            with tape.stop_recording():
                for i in range(self.n_labels):
                    log_prob = log_probs[i]
                    grad = tape.gradient(log_prob, self.variables)
                    sq_grad = [probs[i] * tf.square(g) for g in grad]
                    sq_grads.append(sq_grad)

        # Take the average across logits. The per-logit weight was added
        # earlier as each per-logit square gradient was weighted by the
        # probability of the class according to the output distribution.
        return [tf.reduce_sum(g, axis=0) for g in zip(*sq_grads)], logits

    @tf.function
    def compute_dense_d_pefs_and_logits_for_batch(self, batch):
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
        # Each item's shape is [batch, var_shape]
        return fishers, logits

    @tf.function
    def sparsify_batch_d_pefs(self, flat_batch_pefs: tf.Tensor):
        values, indices = tf.math.top_k(flat_batch_pefs, k=self.tf_n_values_per_example)
        return values, tf.cast(indices, tf.int32)

    @tf.function
    def process_batch(self, batch):
        fishers, logits = self.compute_dense_d_pefs_and_logits_for_batch(batch)

        flat_fishers = flatten_batch_d_pefs(fishers)

        pef_norms = _compute_d_pef_norms(flat_fishers)
        values, indices = self.sparsify_batch_d_pefs(flat_fishers)

        return {
            'logits': logits,
            'pef_norms': pef_norms,
            'values': values,
            'indices': indices,
        }


@dataclasses.dataclass
class TopClassesSparseDPefComputer(AllClassesSparseDPefComputer):
    min_prob_class: float

    def __post_init__(self):
        super().__post_init__()
        if self.vectorized:
            raise ValueError('TopClassesSparseDPefComputer only supports non-vectorized computation.')

        self._fisher_acc_vars = [
            tf.Variable(tf.zeros_like(v), trainable=False)
            for v in self.variables
        ]

    @tf.function
    def _fisher_single_example(self, single_example_batch):
        # single_example_batch is assumed to have the dummy batch dimension of 1.
        for f in self._fisher_acc_vars:
            f.assign(tf.zeros_like(f))

        variables = self.variables

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(variables)
            logits = npeff_models.compute_logits(self.model, single_example_batch)

            # The batch dimension must be 1 to call the model, so we remove it here.
            logits = tf.squeeze(logits, axis=0)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            probs = tf.nn.softmax(logits, axis=-1)

            for i in tf.range(self.n_labels):
                log_prob = log_probs[i]

                with tape.stop_recording():
                    if probs[i] < self.min_prob_class:
                        continue
                    grad = tape.gradient(log_prob, variables)
                    for f, g in zip(self._fisher_acc_vars, grad):
                        f.assign_add(probs[i] * tf.square(g))

        fishers = [tf.identity(f) for f in self._fisher_acc_vars]
        return logits, fishers

    @tf.function
    def compute_dense_d_pefs_and_logits_for_batch(self, batch):
        batch_size = batch_size_from_batch(batch)
        batch = expand_batch_dims(batch)

        logits_ta = tf.TensorArray(tf.float32, size=batch_size, infer_shape=False)
        # NOTE: We are storing the flattened fishers here instead of per-variable.
        fishers_ta = tf.TensorArray(tf.float32, size=batch_size, infer_shape=False)

        for i in tf.range(batch_size):
            ex_fishers, ex_logits = self._fisher_single_example(slice_batch(batch, i))
            logits_ta = logits_ta.write(i, ex_logits)
            fishers_ta = fishers_ta.write(i, tf.concat([tf.reshape(f, [-1]) for f in ex_fishers], axis=0))

        logits = logits_ta.stack()
        fishers = fishers_ta.stack()

        # We return the fishers wrapped in a list because the parent class expects it
        # to be a list of tensors corresponding to each variable. It should be fine if
        # we trick it like this.
        return [fishers], logits

###############################################################################


@dataclasses.dataclass
class StreamingDPefSaver:
    fisher_computer: Union[AllClassesSparseDPefComputer, TopClassesSparseDPefComputer]

    n_examples: int

    use_tqdm: bool = True

    def __post_init__(self):
        self.model = self.fisher_computer.model
        self.variables = self.fisher_computer.variables
        self.n_values_per_example = self.fisher_computer.n_values_per_example

        self.n_classes = int(self.model.num_labels)
        self.n_parameters = int(tf.reduce_sum([tf.size(v) for v in self.variables]).numpy())

        self.n_examples_processed = None

    def _initialize_file(self, file: h5py.File):
        n_examples = self.n_examples
        self.data_grp = file.create_group('data')
        self.fisher_grp = self.data_grp.create_group('fisher')

        self.fisher_grp.attrs['dense_fisher_size'] = self.n_parameters

        int_ds = lambda n, s: self.data_grp.create_dataset(n, s, dtype=np.int32)
        flt_ds = lambda n, s: self.data_grp.create_dataset(n, s, dtype=np.float32)

        self.labels_ds = int_ds('labels', [n_examples])
        self.logits_ds = flt_ds('predicted_logits', [n_examples, self.n_classes])

        self.values_ds = self.fisher_grp.create_dataset('values', [n_examples, self.n_values_per_example], dtype=np.float32)
        self.indices_ds = self.fisher_grp.create_dataset('indices', [n_examples, self.n_values_per_example], dtype=np.int32)

        self.pef_norms_ds = flt_ds('dense_fisher_norms', [n_examples])

    def _write(self, h5_ds: h5py.Dataset, data: tf.Tensor):
        i1 = self.n_examples_processed
        i2 = min(self.n_examples_processed + data.shape[0], self.n_examples)
        h5_ds[i1:i2] = data[:i2 - i1].numpy().astype(h5_ds.dtype)

    def _write_batch_results_to_file(self, file, batch_dict, labels):
        d = batch_dict

        self._write(self.labels_ds, labels)
        self._write(self.logits_ds, d['logits'])

        self._write(self.values_ds, d['values'])
        self._write(self.indices_ds, d['indices'])

        self._write(self.pef_norms_ds, d['pef_norms_ds'])

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
