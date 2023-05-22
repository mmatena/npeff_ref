"""Compute the diagonal approximation of a dataset-level Fisher matrix."""
import os

from absl import app
from absl import flags

from npeff.datasets import npeff_datasets
from npeff.fishers import d_dsf_computer
from npeff.models import npeff_models


COMPUTATION_MODES = ['expectation', 'sampling']


FLAGS = flags.FLAGS

flags.DEFINE_string("output_filepath", None, "Path to h5 file to write output to.")

flags.DEFINE_string("model", None, "String indicating model to use.")

flags.DEFINE_string("task", None, "String indicating dataset to use. See npeff_datasets for more info.")
flags.DEFINE_string("split", None, "Split of dataset to use.")
flags.DEFINE_integer("n_examples", None, "Numbmer of examples to use to compute PEFs.")
flags.DEFINE_integer("batch_size", 4, "Batch size to use when computing PEFs.")

flags.DEFINE_enum('computation_mode', None, COMPUTATION_MODES, 'Whether to sample from the model predictive distribution '
                                                               'for each example or take the exact expectation.')

# HuggingFace model only flags:
flags.DEFINE_bool("from_pt", True, "Whether the model is from PyTorch. For Hugging Face models only.")

# Text task only flags:
flags.DEFINE_integer("sequence_length", 128, "Sequence length to use for transformers.")
flags.DEFINE_string("tokenizer", None, "Tokenizer to use. Defaults to --model if not set for a text task.")


###############################################################################

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


def main(_):
    model_str = os.path.expanduser(FLAGS.model)
    model = npeff_models.from_pretrained(model_str, from_pt=FLAGS.from_pt)

    # NOTE: For transformers, trainable_variables and variables are the same. I am
    # not sure what is correct for vision models.
    variables = model.trainable_variables

    tokenizer = npeff_models.load_tokenizer(FLAGS.tokenizer or model_str)
    ds = create_dataset(tokenizer)

    computer = d_dsf_computer.DDsfComputer(
        model=model,
        variables=variables,
        n_examples=FLAGS.n_examples,
        sample_from_logits=FLAGS.computation_mode == 'sampling',
        use_tqdm=True,
    )

    dsf = computer.compute_fisher(ds)
    dsf.save(FLAGS.output_filepath)


if __name__ == "__main__":
    app.run(main)
