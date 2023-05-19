"""Stuff for evaluating models"""
import dataclasses
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

from npeff.models import npeff_models

# typdefs
NpExamples = Union[np.ndarray, Dict[str, np.ndarray]]

###############################################################################


@dataclasses.dataclass
class EvaluationResults:
    labels: np.ndarray
    logits: np.ndarray

    og_logits: Optional[np.ndarray] = None

    def __post_init__(self):
        self.predictions = np.argmax(self.logits, axis=-1)
        self.correct_predictions = self.predictions == self.labels

    ############################################################

    def acc(self) -> float:
        return self.correct_predictions.astype(np.float64).mean()

    def acc_for_examples(self, example_indices: Sequence[int]) -> float:
        example_indices = np.array(list(sorted(example_indices)), dtype=np.int32)
        return self.correct_predictions[example_indices].astype(np.float64).mean()

    ############################################################

    def _loss(self, labels, logits) -> float:
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True).numpy().mean()

    def loss(self) -> float:
        return self._loss(self.labels, self.logits)

    def loss_for_examples(self, example_indices: Sequence[int]) -> float:
        example_indices = np.array(list(sorted(example_indices)), dtype=np.int32)
        return self._loss(self.labels[example_indices], self.logits[example_indices])

    ############################################################

    def _kl(self, og_logits, logits) -> float:
        assert og_logits is not None
        return tf.keras.losses.kl_divergence(tf.math.softmax(logits), tf.math.softmax(og_logits)).numpy().mean()

    def kl(self) -> float:
        return self._kl(self.og_logits, self.logits)

    def kl_for_examples(self, example_indices: Sequence[int]) -> float:
        example_indices = np.array(list(sorted(example_indices)), dtype=np.int32)
        return self._kl(self.og_logits[example_indices], self.logits[example_indices])


@dataclasses.dataclass
class EvaluationContext:
    examples: Tuple[NpExamples, np.ndarray]
    og_logits: np.ndarray

    batch_size: int = 128

    def get_ds(self, example_indices: Optional[np.ndarray] = None) -> tf.data.Dataset:
        if example_indices is None:
            return tf.data.Dataset.from_tensor_slices(self.examples)
        return tf.data.Dataset.from_tensor_slices(
            slice_examples(self.examples, example_indices))

    def evaluate(self, model, example_indices: Optional[np.ndarray] = None) -> EvaluationResults:
        labels, logits = [], []
        for x, y in self.get_ds(example_indices).batch(self.batch_size):
            labels.append(y.numpy())
            batch_logits = npeff_models.compute_logits(model, x, training=False).numpy()
            logits.append(batch_logits)

        return EvaluationResults(
            labels=np.concatenate(labels, axis=0),
            logits=np.concatenate(logits, axis=0)[:, :self.og_logits.shape[-1]],
            og_logits=self.og_logits[example_indices] if example_indices is not None else self.og_logits,
        )

    def get_og_accuracy(self, example_indices: Optional[np.ndarray] = None):
        labels = self.examples[1]
        logits = self.og_logits

        if example_indices is not None:
            labels = labels[example_indices]
            logits = logits[example_indices]

        preds = np.argmax(logits, axis=-1)
        return (preds == labels).astype(np.float64).mean()

    @classmethod
    def create_from_ds_and_logits(cls, ds: tf.data.Dataset, logits: np.ndarray, **kwargs):
        # The ds should NOT be batched.
        n_examples = logits.shape[0]
        for all_examples in ds.take(n_examples).batch(n_examples).as_numpy_iterator():
            break
        return cls(
            all_examples=all_examples,
            og_logits=logits,
            **kwargs,
        )


###############################################################################


def slice_examples(examples: Tuple[NpExamples, np.ndarray], example_indices: np.ndarray):
    examples, labels = examples
    if isinstance(examples, np.ndarray):
        examples_slice = examples[example_indices]
    else:
        examples_slice = {
            k: v[example_indices]
            for k, v in examples.items()
        }
    return examples_slice, labels[example_indices]
