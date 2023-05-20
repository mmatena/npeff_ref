"""Module-level stuff for the datasets code."""
from typing import Union

import tensorflow as tf

from . import glue
from . import snli
from . import imagenet


###############################################################################


def load(
    task_ri: str,
    split: str,
    tokenizer,
    sequence_length: int,
    *extra_args,
    **extra_kwargs,
) -> tf.data.Dataset:
    group, task = split_task_ri(task_ri)
    return _get_group_module(group).load(
        task=task,
        split=split,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        *extra_args,
        **extra_kwargs,
    )


def load_raw(
    task_ri: str,
    split: str,
    *extra_args,
    **extra_kwargs,
) -> tf.data.Dataset:
    """Loads the raw dataset, essentially directly from tfds.

    Might not be implemented for every dataset.
    """
    group, task = split_task_ri(task_ri)
    return _get_group_module(group).load_raw(
        task=task, split=split, *extra_args, **extra_kwargs)


def n_classes_for_task(task_ri: str) -> int:
    group, task = split_task_ri(task_ri)
    return _get_group_module(group).n_classes_for_task(task)


def de_facto_validation_split(task_ri: str) -> Union[str, None]:
    """Returns the name of the split typically used for validation.

    This is usually the "validation" or "test" split. For tasks without such
    a split, a None is returned.
    """
    group, task = split_task_ri(task_ri)
    return _get_group_module(group).de_facto_validation_split(task)


def examples_per_epoch(task_ri: str) -> Union[str, None]:
    """Returns the number of examples in each epoch of training.
    
    Returns None is this is not supported.
    """
    group, task = split_task_ri(task_ri)
    return _get_group_module(group).examples_per_epoch(task)


###############################################################################

def split_task_ri(task_ri: str):
    # The "ri" stands for resource indicator.
    group, *task = task_ri.split('/')
    return group, '/'.join(task)


def _get_group_module(group: str):
    if group == 'glue':
        return glue
    elif group == 'imagenet':
        return imagenet
    elif group == 'snli':
        return snli
    else:
        raise ValueError(f'Invalid task group: {group}')
