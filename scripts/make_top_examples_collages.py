"""Makes collages of top example images for components of image NPEFFs.

Images in the collages are sorted in descending order of component coefficient, going
from the left to right and top to bottom in a row-major format.
"""
import os

from absl import app
from absl import flags

import matplotlib.pyplot as plt

from npeff.datasets import npeff_datasets
from npeff.decomp import decomps
from npeff.viewers import collage_generator

FLAGS = flags.FLAGS

flags.DEFINE_string("output_directory", None, "Directory where to write collages to. Will be created if it does not exist.")

flags.DEFINE_string("decomposition_filepath", None, "Filepath of NPEFF decomposition.")

flags.DEFINE_string("task", None, "String indicating dataset to use. See npeff_datasets for more info.")
flags.DEFINE_string("split", None, "Split of dataset to use.")

flags.DEFINE_list("component_indices", None, 'Leave set to None to run on all components.')

flags.DEFINE_integer('n_rows', None, 'Number of rows in each collage image.')
flags.DEFINE_integer('n_cols', None, 'Number of cols in each collage image.')

flags.DEFINE_string("image_extension", 'jpg', "Extension of the generated images.")
flags.DEFINE_string("image_filename_prefix", 'comp', "Prefix of filename for each generated collage.")

##########################################################################


def get_component_indices(W):
    if FLAGS.component_indices is None:
        return list(range(W.shape[-1]))
    else:
        return [int(i) for i in FLAGS.component_indices]


def main(_):
    output_directory = os.path.expanduser(FLAGS.output_directory)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    ds = npeff_datasets.load(task=FLAGS.task, split=FLAGS.split)
    W = decomps.load_W(FLAGS.decomposition_filepath)

    generator = collage_generator.TopExamplesCollageGenerator(
        W=W, ds=ds, n_rows=FLAGS.n_rows, n_cols=FLAGS.n_cols)

    for comp_index in get_component_indices():
        img = generator.make_collage(comp_index)
        filename = f'{FLAGS.image_filename_prefix}{comp_index}.{FLAGS.image_extension}'
        plt.imsave(os.path.join(output_directory, filename, img))


if __name__ == "__main__":
    app.run(main)
