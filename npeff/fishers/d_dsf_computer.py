"""Code for computing diagonal approximations to dataset-level fishers."""
import dataclasses
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from npeff.models import npeff_models

from . import dsfs

tfd = tfp.distributions


###############################################################################


def expand_batch_dims(batch):
    if isinstance(batch, tf.Tensor):
        return tf.expand_dims(batch, axis=1)
    else:
        return {k: tf.expand_dims(v, axis=1) for k, v in batch.items()}


###############################################################################


@dataclasses.dataclass
class DDsfComputer:
    model: tf.keras.Model
    variables: List[tf.Variable]

    n_examples: int

    sample_from_logits: bool = False

    use_tqdm: bool = True

    def __post_init__(self):
        self.num_labels = self.model.num_labels

        self.fishers = [tf.Variable(tf.zeros_like(v)) for v in self.variables]

    @tf.function
    def _fisher_single_example_expected(self, single_example_batch):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.variables)

            logits = npeff_models.compute_logits(single_example_batch, training=False)

            # The batch dimension must be 1 to call the model, so we remove it
            # here.
            logits = tf.squeeze(logits, axis=0)

            log_probs = tf.nn.log_softmax(logits, axis=-1)
            probs = tf.nn.softmax(logits, axis=-1)

            sq_grads = []
            log_probs = [log_probs[i] for i in range(self.num_labels)]
            with tape.stop_recording():
                for i in range(self.num_labels):
                    log_prob = log_probs[i]
                    grad = tape.gradient(log_prob, self.variables)
                    sq_grad = [probs[i] * tf.square(g) for g in grad]
                    sq_grads.append(sq_grad)
        # Take the average across logits. The per-logit weight was added
        # earlier as each per-logit square gradient was weighted by the
        # probability of the class according to the output distribution.
        return [tf.reduce_sum(g, axis=0) for g in zip(*sq_grads)]

    @tf.function
    def _fisher_single_example_sampled(self, single_example_batch):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.variables)

            logits = npeff_models.compute_logits(single_example_batch, training=False)

            # The batch dimension must be 1 to call the model, so we remove it here.
            logits = tf.squeeze(logits, axis=0)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            chosen_index = tfd.Categorical(logits=logits).sample()
            log_prob = log_probs[chosen_index]

        grad = tape.gradient(log_prob, self.variables)
        return [tf.square(g) for g in grad]

    @tf.function
    def _process_batch(self, batch):
        batch = expand_batch_dims(batch)
        if self.sample_from_logits:
            fisher_single_example_fn = self._fisher_single_example_sampled
        else:
            fisher_single_example_fn = self._fisher_single_example_expected

        fishers = tf.vectorized_map(fisher_single_example_fn, batch)

        for f, b in zip(self.fishers, fishers):
            f.assign_add(tf.reduce_sum(b, axis=0) / float(self.n_examples))

    def compute_fisher(self, ds: tf.data.Dataset) -> dsfs.DenseDDsf:
        # The dataset SHOULD be batched and FINITE.
        if self.use_tqdm:
            ds = tqdm(ds)

        # Reset the fisher accumulators.
        for f in self.fishers:
            f.assign(tf.zeros_like(f))

        for x, _ in ds:
            self._process_batch(x)

        return dsfs.DenseDDsf(
            fishers=np.concatenate([
                tf.reshape(f, [-1]).numpy()
                for f in self.fishers
            ], axis=0),
        )
