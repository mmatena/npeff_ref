"""Runs perturbation experiments for L(V)RM components.

For each component, this script will print:
    - Maximum perturbed model KL-divergence ratio.
    - Maximum perturbed model baseline accuracy.
    - Maximum perturbed model top components accuracy.
    - Original model top components accuracy.

This script will also print out the baseline accuracy of
the original model.
"""
import os

from absl import app
from absl import flags
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

from npeff.datasets import npeff_datasets
from npeff.decomp import decomps
from npeff.fishers import pefs
from npeff.models import npeff_models
from npeff.perturbations import evaluation
from npeff.perturbations import lrm_perturbations


# The NLI_RELABEL is used to address a mismatch between the labels used
# in our SNLI and MNLI datasets and the labels used by some saved
# Hugging Face models.
SPECIAL_PROCESSING_TYPES = [None, 'NLI_RELABEL']


FLAGS = flags.FLAGS

flags.DEFINE_string("decomposition_filepath", None, "Filepath of L(V)RM-NPEFF decomposition.")
# TODO: Allow --pef_filepath to be None and compute the logits within this script.
flags.DEFINE_string("pef_filepath", None, "Filepath of L(V)RM-PEFs. We only use the logits from this file.")

flags.DEFINE_string("model", None, "String indicating model to use.")

flags.DEFINE_string("task", None, "String indicating dataset to use. See npeff_datasets for more info.")
flags.DEFINE_string("split", None, "Split of dataset to use.")
flags.DEFINE_integer("n_examples", None, "Number of examples to use to compute PEFs.")
flags.DEFINE_integer("batch_size", 64, "Batch size to use when evaluating perturbed models.")

flags.DEFINE_list("component_indices", None, 'Leave set to None to run on all components.')

flags.DEFINE_integer("n_top_examples", None, 'Use the top this many examples from each component.')
flags.DEFINE_integer("n_baseline_examples", None, 'Use this many examples as the baseline set.')

flags.DEFINE_float("perturbation_magnitude", None, 'L2 magnitude of the perturbation.')
flags.DEFINE_float("max_abs_cos_sim", None,
                   'Max abs cosine similarity between component pseudo-Fisher vectors to perform orthogonal rejection. '
                   'Leave None to not do the semi-orthogolization.')

flags.DEFINE_enum("special_processing", None, SPECIAL_PROCESSING_TYPES, "Optional. Special processing to apply.")


# HuggingFace model only flags:
flags.DEFINE_bool("from_pt", True, "Whether the model is from PyTorch. For Hugging Face models only.")

# Text task only flags:
flags.DEFINE_integer("sequence_length", 128, "Length of sequence to use for transformer models.")
flags.DEFINE_string("tokenizer", None, "Hugging Face tokenizer to use. Defaults to --model if not set for a text task.")


##########################################################################


def create_dataset(tokenizer):
    ds = npeff_datasets.load(
        FLAGS.task,
        split=FLAGS.split,
        tokenizer=tokenizer,
        sequence_length=FLAGS.sequence_length,
    )
    ds = ds.take(FLAGS.n_examples).cache()
    return ds


def get_component_indices(decomp):
    if FLAGS.component_indices is None:
        component_indices = list(range(decomp.W.shape[-1]))
    else:
        component_indices = [int(c) for c in FLAGS.component_indices]

    assert len(component_indices) > 0

    return component_indices


def evaluate_baseline_original_model_acc(helper):
    # TODO: I don't need to call the model to do this. Add method on the eval ctx
    # to get original acc.
    acc = helper.eval_ctx.get_og_accuracy(np.arange(FLAGS.n_baseline_examples))
    print(f'Original model baseline examples accuracy: {acc}')


def print_component_results(perturber, results):
    print(f'Component {perturber.component_index}')
    print(f'    KL-Ratio: {results.max_kl_ratio()}')
    print(f'    Perturbed Baseline Ex Acc: {results.max_baseline_examples_acc()}')
    print(f'    Perturbed Component Ex Acc: {results.max_top_examples_acc()}')
    print(f'    Original Component Ex Acc: {perturber.eval_ctx.get_og_accuracy(perturber.top_inds)}')


def main(_):
    logits = pefs.load_logits(FLAGS.pef_filepath)

    model_str = os.path.expanduser(FLAGS.model)
    model = npeff_models.from_pretrained(model_str, from_pt=FLAGS.from_pt)

    # NOTE: For transformers, trainable_variables and variables are the same. I am
    # not sure what is correct for vision models.
    variables = model.trainable_variables

    tokenizer = npeff_models.load_tokenizer(FLAGS.tokenizer or model_str)
    ds = create_dataset(tokenizer)

    eval_ctx = evaluation.EvaluationContext.create_from_ds_and_logits(
        ds=ds, logits=logits, batch_size=FLAGS.batch_size)

    if FLAGS.special_processing == 'NLI_RELABEL':
        eval_ctx.all_examples = (eval_ctx.all_examples[0], (eval_ctx.all_examples[1] + 1) % 3)

    decomp = decomps.LrmNpeffDecomposition.load(FLAGS.decomposition_filepath)

    helper = lrm_perturbations.LrmPerturbationHelper(
        decomp=decomp,
        model=model,
        variables=variables,
        eval_ctx=eval_ctx,
        n_top_examples=FLAGS.n_top_examples,
        n_baseline_examples=FLAGS.n_baseline_examples,
    )

    evaluate_baseline_original_model_acc(helper)

    for comp_index in get_component_indices(decomp):
        perturber = lrm_perturbations.LrmComponentPerturber(
            helper=helper, component_index=comp_index, max_sim=FLAGS.max_abs_cos_sim)
        results = perturber.evaluate_pm(FLAGS.perturbation_magnitude)
        print_component_results(perturber, results)


if __name__ == "__main__":
    app.run(main)
