"""Code for the SNLI dataset."""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from . import glue

###############################################################################

SNLI_TASK_NAMES = ('default',)

ARTIFICIAL_SPLITS = ('train_skip_50k',)

###############################################################################


def load(
    task: str,
    split: str,
    tokenizer,
    sequence_length: int,
):
    ds = load_raw(task, split)
    ds = ds.map(_to_mnli_style)
    ds = glue.convert_dataset_to_features(
        ds,
        tokenizer,
        sequence_length,
        task='mnli',
    )
    return ds


def load_raw(task: str, split: str):
    if task not in SNLI_TASK_NAMES:
        raise ValueError(f'Invalid snli task: {task}')

    ds = tfds.load("snli", split=_to_tfds_split(split))

    if split == 'train_skip_50k':
        ds = ds.skip(50_000)

    return ds


def n_classes_for_task(task: str):
    return 3


def de_facto_validation_split(task):
    return 'validation'


def examples_per_epoch(task):
    return 550_152


###############################################################################


def _to_tfds_split(split: str) -> str:
    if split == "train_skip_50k":
        return "train"
    return split


def _to_mnli_style(x):
    x['idx'] = tf.constant(0, dtype=tf.int32)
    return x
