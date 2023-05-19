"""Stuff for dealing with models."""
import tensorflow as tf

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

from . import resnet

DEFAULT_MODEL_GROUP = "transformer"

###############################################################################


def from_pretrained(model_ri: str, *, from_pt: bool = True, **kwargs):
    group, model = split_model_ri(model_ri)

    if group == 'transformer':
        return TFAutoModelForSequenceClassification.from_pretrained(model, from_pt=from_pt, **kwargs)

    elif group == 'resnet':
        return resnet.from_pretrained(model, **kwargs)

    else:
        raise ValueError(f'Invalid model group: {group}')


def load_tokenizer(model_ri: str):
    group, model = split_model_ri(model_ri)
    if group == 'transformer':
        return AutoTokenizer.from_pretrained(model)
    else:
        return None


###############################################################################


def compute_logits(model, batch, *, training=False):
    logits = model(batch, training=training)
    if not isinstance(logits, tf.Tensor):
        # This should happen for HuggingfaceModels
        logits = logits.logits
    return logits


###############################################################################


def split_model_ri(model_ri: str):
    # The "ri" stands for resource indicator.
    splits = model_ri.split(':')
    if len(splits) == 1:
        return DEFAULT_MODEL_GROUP, model_ri
    group, model = splits
    return group, model


def is_image_model(model_ri: str) -> bool:
    return split_model_ri(model_ri)[0] == 'resnet'
