"""Convenience wrapper around resnet."""
import tensorflow as tf

###############################################################################

RESNET_NAMES = ('resnet50_imagenet',)

###############################################################################


def from_pretrained(model_str: str):
    assert model_str in RESNET_NAMES

    if model_str == 'resnet50_imagenet':
        model = tf.keras.applications.resnet50.ResNet50(
            include_top=True,
            weights='imagenet',
            classes=1000,
            classifier_activation=None,
        )
        setattr(model, 'num_labels', 1000)
        model.compile()
        return model

    else:
        raise ValueError(model_str)
