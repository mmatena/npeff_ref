"""Utilities for going from a list of arrays/Tensors to a single array/Tensor and back again."""
import dataclasses
import functools
from typing import List, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

# typedefs
SparseTensor = tf.sparse.SparseTensor
TensorOrArray = Union[tf.Tensor, np.ndarray]


@dataclasses.dataclass
class FlatPacker:
    """Store several tensors/arrays in a single flat tensor/arrays.

    TODO: Ensure this handles scalars properly.
    """

    # Will be converted to Tuple[tf.TensorShape, ...] in __post_init__.
    tensor_shapes: Sequence[Union[tf.TensorShape, Sequence[int]]]

    def __post_init__(self):
        # Ensure consistent type.
        self.tensor_shapes = tuple(tf.TensorShape(s) for s in self.tensor_shapes)

        self._tensor_sizes = [s.num_elements() for s in self.tensor_shapes]
        assert all(s is not None for s in self._tensor_sizes)

        self._offsets = [0]
        for s in self._tensor_sizes[:-1]:
            self._offsets.append(self._offsets[-1] + s)

        self.flat_size = self._offsets[-1] + self._tensor_sizes[-1]

    def get_range_for_tensor_by_index(self, tensor_index: int) -> Tuple[int, int]:
        start = self._offsets[tensor_index]
        end = start + self._tensor_sizes[tensor_index]
        return start, end

    def encode_tf(self, tensors: Sequence[TensorOrArray]) -> tf.Tensor:
        # First do a bunch of validation.
        assert len(tensors) == len(self.tensor_shapes)

        all_batch_dims = set()
        for s, t in zip(self.tensor_shapes, tensors):
            assert t.shape[-len(s):] == s
            batch_dims = tuple(t.shape[:-len(s)])
            all_batch_dims.add(batch_dims)

        assert len(all_batch_dims) == 1, 'Inconsistent batch dimensions'

        # Done with validation, now encode.
        batch_shape, = list(all_batch_dims)
        batch_shape = tf.TensorShape(batch_shape)

        flatten_shape = batch_shape.as_list() + [-1]
        return tf.concat(
            [
                tf.reshape(t, flatten_shape)
                for t in tensors
            ],
            axis=-1)

    def decode_tf(self, tensor: TensorOrArray) -> List[tf.Tensor]:
        assert tensor.shape[-1] == self.flat_size

        batch_shape = tf.TensorShape(tensor.shape[:-1])

        decoded = []
        for offset, shape in zip(self._offsets, self.tensor_shapes):
            dec = tensor[..., offset : offset + shape.num_elements()]
            dec = tf.reshape(dec, batch_shape.concatenate(shape))
            decoded.append(dec)

        return decoded

    def decode_sparse_tf(self, values: TensorOrArray, indices: TensorOrArray) -> List[SparseTensor]:
        # The values argument supports some batching. The indices currently does not.
        values = tf.cast(values, tf.float32)
        indices = tf.cast(indices, tf.int64)

        if len(indices.shape) != 1:
            indices = tf.squeeze(indices, axis=-1)
        assert len(indices.shape) == 1

        assert values.shape[-1] == indices.shape[-1]
        assert tf.reduce_max(indices) <= self.flat_size

        batch_shape = tf.TensorShape(values.shape[:-1])

        decoded = []
        for offset, shape in zip(self._offsets, self.tensor_shapes):
            dec_mask = (offset <= indices) & (indices < offset + shape.num_elements())

            dec_indices = tf.boolean_mask(indices, dec_mask) - offset
            dec_values = _tf_boolean_mask_second_axis(values, dec_mask)

            dec = tf.sparse.SparseTensor(
                indices=_flat_to_sparse_indices(dec_indices, batch_shape),
                values=tf.reshape(dec_values, [-1]),
                dense_shape=batch_shape.concatenate([shape.num_elements()])
            )
            dec = tf.sparse.reshape(dec, batch_shape.concatenate(shape))
            decoded.append(dec)

        return decoded

    def convert_global_indices_to_flat_per_tensor_indices(self, indices: TensorOrArray) -> List[tf.Tensor]:
        # No support of batching.
        indices = tf.cast(indices, tf.int64)
        assert len(indices.shape) == 1

        ret = []
        for offset, shape in zip(self._offsets, self.tensor_shapes):
            offset_end = offset + shape.num_elements()
            mask = (offset <= indices) & (indices < offset_end)
            ret.append(tf.boolean_mask(indices, mask) - offset)

        return ret


@tf.function(experimental_relax_shapes=True)
def _tf_boolean_mask_second_axis(values, dec_mask):
    # Need to create this function to prevent retracing of the vectorized map in the loop.
    return tf.vectorized_map(_tf_boolean_mask_vec_map_fn, (values, tf.expand_dims(dec_mask, axis=0)))


@tf.function(experimental_relax_shapes=True)
def _tf_boolean_mask_vec_map_fn(arg):
    return tf.boolean_mask(arg[0], arg[1])


def _flat_to_sparse_indices(flat_indices: tf.Tensor, batch_shape: tf.TensorShape):
    assert flat_indices.shape.rank == 1
    indices_length, = flat_indices.shape

    batch_rank = batch_shape.rank
    assert batch_rank <= 1, 'TODO: Support more than 1 batch dimension.'

    if batch_rank == 0:
        return tf.expand_dims(flat_indices, axis=-1)

    expanded_inds = tf.ones(batch_shape.concatenate([indices_length]), dtype=flat_indices.dtype)
    expanded_inds *= flat_indices

    first_index = tf.ones_like(expanded_inds) * tf.range(batch_shape[0], dtype=flat_indices.dtype)[:, None]
    return tf.stack([
        tf.reshape(first_index, [-1]),
        tf.reshape(expanded_inds, [-1]),
    ], axis=-1)
