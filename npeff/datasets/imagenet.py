"""ImageNet"""
import functools
from typing import List, Optional, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

###############################################################################

IMAGENET_TASK_NAMES = ('default', 'resnet')

DEFAULT_IMAGE_SIZE = 224

###############################################################################


def load(
    task: str,
    split: str,
    tokenizer=None,
    sequence_length: int = None,
):
    del tokenizer, sequence_length

    image_size = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

    if task not in IMAGENET_TASK_NAMES:
        raise ValueError(f'Invalid ImageNet task: {task}')

    ds = tfds.load("imagenet2012", split=split)
    ds = ds.map(functools.partial(_first_preprocess_fn, image_size=image_size))
    if task == 'resnet':
        ds = ds.map(_resnet_preprocess_fn)
    return ds


def load_raw(task: str, split: str):
    raise NotImplementedError


def n_classes_for_task(task: str):
    return 1000


def de_facto_validation_split(task):
    return 'validation'


def examples_per_epoch(task):
    return None


###############################################################################

def _first_preprocess_fn(entry, image_size: Tuple[int, int]):
    x = tf.cast(entry['image'], tf.float32)
    x = tf.image.resize(x, image_size)

    y = entry['label']

    return x, y


def _resnet_preprocess_fn(x, y):
    x = tf.keras.applications.resnet50.preprocess_input(x)
    return x, y


###############################################################################

def get_image_filenames(split: str, n_examples: Optional[int] = None) -> List[str]:
    filenames = tfds.load("imagenet2012", split=split).map(lambda x: x['file_name'])
    if n_examples is not None:
        filenames = filenames.take(n_examples)
    return [tf.compat.as_str(f) for f in filenames.as_numpy_iterator()]
