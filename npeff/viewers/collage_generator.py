"""Generates collage of top component images."""
import dataclasses

import numpy as np
import tensorflow as tf


@dataclasses.dataclass
class TopExamplesCollageGenerator:
    # NPEFF coefficients.
    # W.shape = [n_npeff_examples, n_components]
    W: np.ndarray

    # Dataset of labelled images.
    ds: tf.data.Dataset

    # Number of rows and columns to include in the collage.
    n_rows: int
    n_cols: int

    def __post_init__(self):
        self.n_npeff_examples, self.n_components = self.W.shape

        self.n_examples = self.n_rows * self.n_cols

        # image.shape = [n_npeff_examples, image_size, image_size, 3]
        self.images = self._make_images()

        assert self.images.shape[1] == self.images.shape[2]
        self.image_size = self.images.shape[1]

    def _make_images(self):
        for x, _, in self.ds.batch(self.n_npeff_examples).as_numpy_iterator():
            break
        return x

    def make_collage(self, component_index: int):
        Q = self.image_size
        top_inds = np.argsort(-self.W[:, component_index])[:self.n_examples]

        ret = np.zeros([self.n_rows * Q, self.n_cols * Q, 3], dtype=np.float64)

        for i, ind in enumerate(top_inds):
            col = i % self.n_cols
            row = i // self.n_cols
            ret[row * Q : (row + 1) * Q, col * Q : (col + 1) * Q, :] = self.images[ind]

        return ret
