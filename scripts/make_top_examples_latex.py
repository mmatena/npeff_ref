R"""Makes a latex representation of the top examples of components.

Recommended to XeLaTex when compiling to pdf to handle unicode properly.
"""
import os

from absl import app
from absl import flags

from npeff.datasets import npeff_datasets
from npeff.decomp import decomps
from npeff.fishers import pefs
from npeff.viewers import latex_generator

# The NLI_RELABEL is used to address a mismatch between the labels used
# in our SNLI and MNLI datasets and the labels used by some saved
# Hugging Face models.
SPECIAL_PROCESSING_TYPES = [None, 'NLI_RELABEL']


FLAGS = flags.FLAGS

flags.DEFINE_string("output_filepath", None, "Filepath of the tex file that will be created.")

flags.DEFINE_string("decomposition_filepath", None, "Filepath of NPEFF decomposition.")
flags.DEFINE_string("pef_filepath", None, "Filepath of PEFs. We only use the logits from this file.")

flags.DEFINE_string("task", None, "String indicating dataset to use. See npeff_datasets for more info.")
flags.DEFINE_string("split", None, "Split of dataset to use.")

flags.DEFINE_list("component_indices", None, 'Leave set to None to run on all components.')
flags.DEFINE_integer('n_examples', None, 'Number of top examples to include for each component.')

flags.DEFINE_string("label_key", 'label', 'Key from the TFDS containing the label.')
flags.DEFINE_list("example_keys", None, 'List of keys from the TFDS dict to use to represent component. '
                                        'The sentences corresponding to each key will be displayed for '
                                        'each example in the order listed in this flag.')

flags.DEFINE_list('label_names', None, 'Names to use to represent each label. '
                                       'In the data should be an integer indexing into this list.')
flags.DEFINE_list('label_colors', None, 'Optional color to use for each label. By default, we use the dvipsnames '
                                        'xcolor package. If present, should be parallel list to --label_names. If '
                                        'not set, then all labels will just be black.')

flags.DEFINE_string("fontsize", 'footnotesize', "LaTeX font-size to size.")

flags.DEFINE_enum("special_processing", None, SPECIAL_PROCESSING_TYPES, "Optional. Special processing to apply.")


##########################################################################


def _nli_relabel_map_fn(x):
    ret = x.copy()
    ret[FLAGS.label_key] = (ret[FLAGS.label_key] + 1) % 3
    return ret


##########################################################################


def main(_):
    ds = npeff_datasets.load_raw(task=FLAGS.task, split=FLAGS.split)
    if FLAGS.special_processing == 'NLI_RELABEL':
        ds = ds.map(_nli_relabel_map_fn)

    W = decomps.load_W(FLAGS.decomposition_filepath)
    logits = pefs.load_logits(FLAGS.pef_filepath)

    generator = latex_generator.TopExamplesLatexGenerator(
        W=W,
        logits=logits,
        raw_ds=ds,
        n_examples=FLAGS.n_examples,
        label_key=FLAGS.label_key,
        example_keys=FLAGS.example_keys,
        fontsize=FLAGS.fontsize,
        label_names=FLAGS.label_names,
        label_colors=FLAGS.label_colors,
    )

    content = generator.make_latex(FLAGS.component_indices)
    with open(os.path.expanduser(FLAGS.output_filepath), 'wt') as f:
        f.write(content)


if __name__ == "__main__":
    app.run(main)
