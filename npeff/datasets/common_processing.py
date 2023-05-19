"""Common processing for datasets."""
from typing import Dict
import tensorflow as tf


def pad_unbatched_tokenized_examples(
    ds: tf.data.Dataset,
    tokenizer,
    sequence_length: int,
):
    # I think the tokenized examples MUST have length less than or
    # equal to sequence_length for this to work.
    pad_token = tokenizer.pad_token_id
    pad_token_type_id = tokenizer.pad_token_type_id

    def pad_fn(x, y):
        input_ids = x['input_ids']
        padding_length = sequence_length - tf.shape(input_ids)[-1]

        ret = x.copy()
        ret['input_ids'] = tf.concat(
            [input_ids, pad_token * tf.ones(padding_length, dtype=tf.int32)], axis=-1
        )
        # We do this to ensure that the shape is known as this is often
        # needed for downstream steps.
        ret['input_ids'] = tf.reshape(ret['input_ids'], [sequence_length])

        if x.get('token_type_ids', None) is not None:
            ret['token_type_ids'] = tf.concat(
                [
                    x['token_type_ids'],
                    pad_token_type_id * tf.ones(padding_length, dtype=tf.int32),
                ],
                axis=-1,
            )
            ret['token_type_ids'] = tf.reshape(ret['token_type_ids'], [sequence_length])
        return ret, y

    return ds.map(pad_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def pad_unbatched_token_ids(
    input_ids,
    token_type_ids,
    tokenizer,
    sequence_length: int,
):
    # I think the tokenized examples MUST have length less than or
    # equal to sequence_length for this to work.
    pad_token = tokenizer.pad_token_id
    pad_token_type_id = tokenizer.pad_token_type_id

    padding_length = sequence_length - tf.shape(input_ids)[-1]

    input_ids = tf.concat(
        [input_ids, pad_token * tf.ones(padding_length, dtype=tf.int32)], axis=-1
    )
    # We do this to ensure that the shape is known as this is often
    # needed for downstream steps.
    input_ids = tf.reshape(input_ids, [sequence_length])

    if token_type_ids is not None:
        token_type_ids = tf.concat(
            [
                token_type_ids,
                pad_token_type_id * tf.ones(padding_length, dtype=tf.int32),
            ],
            axis=-1,
        )
        token_type_ids = tf.reshape(token_type_ids, [sequence_length])

    return input_ids, token_type_ids


###############################################################################

def trim_batch(x: Dict[str, tf.Tensor]):
    keep_mask = tf.reduce_any(tf.cast(x['attention_mask'], tf.bool), axis=0)

    for k, v in x.items():
        if tf.rank(tf.shape(v)) == 2:
            x[k] = tf.boolean_mask(v, keep_mask, axis=0)

    return x
