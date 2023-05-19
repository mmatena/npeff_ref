"""Computes and saves PEFs to an h5 file."""
import os

from absl import app
from absl import flags

from npeff.datasets import npeff_datasets
from npeff.fishers import d_pefs_computer
from npeff.fishers import lrm_pefs_computer
from npeff.fishers import lvrm_pefs_computer
from npeff.models import npeff_models


PEF_FLAVORS = ['D_PEF', 'LRM_PEF', 'LVRM_PEF']


FLAGS = flags.FLAGS

flags.DEFINE_string("output_filepath", None, "Path to h5 file to write output to.")

FLAGS.DEFINE_enum('pef_type', None, PEF_FLAVORS, 'Type of PEF to compute.')

flags.DEFINE_string("model", None, "String indicating model to use.")

flags.DEFINE_string("task", None, "String indicating dataset to use. See npeff_datasets for more info.")
flags.DEFINE_string("split", None, "Split of dataset to use.")
flags.DEFINE_integer("n_examples", None, "Numbmer of examples to use to compute PEFs.")
flags.DEFINE_integer("batch_size", 4, "Batch size to use when computing PEFs.")

flags.DEFINE_integer("n_fisher_values_per_example", None, "Number of non-zero entries to keep for each example.")


# HuggingFace model only flags:
flags.DEFINE_bool("from_pt", True, "Whether the model is from PyTorch. For Hugging Face models only.")

# Text task only flags:
flags.DEFINE_integer("sequence_length", 128, "Sequence length to use for transformers.")
flags.DEFINE_string("tokenizer", None, "Tokenizer to use. Defaults to --model if not set for a text task.")

# Required for LVRM_PEF. Optional for D_PEF, if left unset it will use all classes. Must be
# unset for LRM_PEF.
flags.DEFINE_float("min_prob_class", None, "Minimum probability assigned to class to use for computing PEFs.")

# Required for LVRM_PEF. Must be unset for all other classes.
flags.DEFINE_integer("max_classes", None, "Maximum number of classes to use when computing a PEF.")


###############################################################################

def _some_flag_validation():
    if FLAGS.pef_type == 'D_PEF':
        if FLAGS.max_classes is not None:
            raise ValueError('The --max_classes flag must be set to None when --pef_type=D_PEF regardless if --min_prob_class is set.')
    elif FLAGS.pef_type == 'LRM_PEF':
        if FLAGS.min_prob_class is not None or FLAGS.max_classes is not None:
            raise ValueError('The --min_prob_class and -max_classes flag must be set to None when --pef_type=LRM_PEF.')
    elif FLAGS.pef_type == 'LVRM_PEF':
        if FLAGS.min_prob_class is None or FLAGS.max_classes is None:
            raise ValueError('The --min_prob_class and -max_classes flag must be set to a value when --pef_type=LVRM_PEF.')


def create_dataset(tokenizer):
    ds = npeff_datasets.load(
        FLAGS.task,
        split=FLAGS.split,
        tokenizer=tokenizer,
        sequence_length=FLAGS.sequence_length,
    )
    ds = ds.take(FLAGS.n_examples).cache()
    ds = ds.batch(FLAGS.batch_size)
    return ds


def create_saver(model, variables):
    if FLAGS.pef_type == 'D_PEF':
        if FLAGS.min_prob_class is None:
            computer = d_pefs_computer.AllClassesSparseDPefComputer(
                model=model,
                variables=variables,
                n_values_per_example=FLAGS.n_fisher_values_per_example,
            )
        else:
            computer = d_pefs_computer.SparseLrmPefComputer(
                model=model,
                variables=variables,
                n_values_per_example=FLAGS.n_fisher_values_per_example,
                min_prob_class=FLAGS.min_prob_class,
            )
        return d_pefs_computer.StreamingDPefSaver(
            fisher_computer=computer,
            n_examples=FLAGS.n_examples,
        )

    elif FLAGS.pef_type == 'LRM_PEF':
        computer = lrm_pefs_computer.SparseLrmPefComputer(
            model=model,
            variables=variables,
            n_values_per_example=FLAGS.n_fisher_values_per_example,
        )
        return lrm_pefs_computer.StreamingLrmPefSaver(
            fisher_computer=computer,
            n_examples=FLAGS.n_examples,
        )

    elif FLAGS.pef_type == 'LVRM_PEF':
        computer = lvrm_pefs_computer.SparseLvrmPefComputer(
            model=model,
            variables=variables,
            n_values_per_example=FLAGS.n_fisher_values_per_example,
            min_prob_class=FLAGS.min_prob_class,
            max_classes=FLAGS.max_classes,
        )
        return lvrm_pefs_computer.LvrmPefSaver(
            fisher_computer=computer,
            n_examples=FLAGS.n_examples,
        )

    else:
        raise ValueError(f'Invalid --pef_type: {FLAGS.pef_type}')

###############################################################################


def main(_):
    _some_flag_validation()

    model_str = os.path.expanduser(FLAGS.model)
    model = npeff_models.from_pretrained(model_str, from_pt=FLAGS.from_pt)

    # NOTE: For transformers, trainable_variables and variables are the same. I am
    # not sure what is correct for vision models.
    variables = model.trainable_variables

    tokenizer = npeff_models.load_tokenizer(FLAGS.tokenizer or model_str)
    ds = create_dataset(tokenizer)

    saver = create_saver(model, variables)

    # TODO: Maybe add some safety in case n_examples is greater than the size of the dataset?
    output_filepath = os.path.expanduser(FLAGS.output_filepath)
    saver.compute_and_save_pefs(output_filepath, ds)


if __name__ == "__main__":
    app.run(main)
