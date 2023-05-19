"""Filters a saved set of PEFs."""
import os

from absl import app
from absl import flags
import numpy as np

from npeff.fishers import pefs

from npeff.util.color_util import cu


# The NLI_RELABEL is used to address a mismatch between the labels used
# in our SNLI and MNLI datasets and the labels used by some saved
# Hugging Face models.
SPECIAL_PROCESSING_TYPES = [None, 'NLI_RELABEL']


FILTER_TYPES = ['INCORRECTS_ONLY']


FLAGS = flags.FLAGS

flags.DEFINE_string("pef_filepath", None, "")
flags.DEFINE_string("output_filepath", None, "Filepath of h5 write to the filtered PEFs to.")

flags.DEFINE_enum("filter_type", 'INCORRECTS_ONLY', FILTER_TYPES, "Type of filtering to perform. Currently only INCORRECTS_ONLY is supported.")

flags.DEFINE_enum("special_processing", None, SPECIAL_PROCESSING_TYPES, "Optional. Special processing to apply.")


###############################################################################

def get_kept_example_indices(og_pefs) -> np.ndarray:
    labels = og_pefs.labels
    predictions = np.argmax(og_pefs.logits, axis=-1)

    if FLAGS.special_processing == 'NLI_RELABEL':
        labels = (labels + 1) % 3

    wrong_pred_inds, = np.nonzero(labels != predictions)

    # Print this so that we have a sanity check that we selected the correct
    # special processing.
    correct_frac = 1 - wrong_pred_inds.shape[0] / labels.shape[0]
    print(cu.hlg(f"Fraction of correct predictions: {correct_frac}"))

    return wrong_pred_inds


def main(_):
    if FLAGS.filter_type != 'INCORRECTS_ONLY':
        raise ValueError(f'Only INCORRECTS_ONLY filtering is supported. Invalid --filter_type flag: {FLAGS.filter_type}')

    pef_filepath = os.path.expanduser(FLAGS.pef_filepath)

    if pefs.infer_pefs_file_class(pef_filepath) == pefs.SparseLvrmPefs:
        raise NotImplementedError('Filtering of LVRM-PEFs is not currently supported.')

    og_pefs = pefs.load(pef_filepath)

    keep_inds = get_kept_example_indices(og_pefs)
    print(cu.hlg(f"Creating PEFS with {keep_inds.shape[0]} examples."))
    filtered_pefs = og_pefs.create_for_subset(keep_inds)

    filtered_pefs.save(os.path.expanduser(FLAGS.output_filepath))


if __name__ == "__main__":
    app.run(main)
