"""Stuff for generating latex for NPEFFs on text models."""
import dataclasses
import re
from typing import Optional, List

import numpy as np
import tensorflow as tf

from . import latex_util


# Make padded label names.
# Do stuff like i did for the manual paper.


##########################################################################

_NUMBERS = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')


def _commandify_name(s: str) -> str:
    """Turn s into something that can be used as a latex command name."""

    # Remove all non-alphanumeric characters.
    s = re.sub(r'\W', '', s)

    # Remove underscores.
    s = s.replace('_', '')

    # Replace numbers with their English text equivalents.
    for n, r in enumerate(_NUMBERS):
        s = s.replace(f'{n}', r)

    return s


_EXAMPLE_CMD_DEF_TEMPLATE = R"""
\newcommand{\compex}[@]{%
\noindent\texttt{[LABEL] #1 {} [PRED] #2 {} [COEFF] #3 \vspace{0mm} \\
$
}}"""

_ENV_TEMPLATE = R"""
\newenvironment{npeffcomp}[1]{%
\subsection{Component #1}
\begin{@}%
}{%
\end{@}%
}"""


_COMPONENTS_LATEX_FILE_START_TEMPLATE = R"""% Please use XeLaTex to handle unicode properly.
\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry} 
\usepackage[dvipsnames]{xcolor}
\usepackage{bold-extra}

% Auto-generated latex commands and environments.
@

\begin{document}

\section{Component Top Examples}

"""

_COMPONENTS_LATEX_FILE_END = R'\end{document}'

##########################################################################


@dataclasses.dataclass
class Example:
    sentences: List[str]
    label: int


@dataclasses.dataclass
class TopExampleLatexGenerator:
    # NPEFF coefficients.
    # W.shape = [n_npeff_examples, n_components]
    W: np.ndarray

    # The model's predictions on the examples.
    # logits.shape = [n_npeff_examples, n_classes]
    logits: np.ndarray

    # Output of calling npeff_datasets.load_raw. Essentially just the
    # tfds dataset. Should NOT be batched.
    raw_ds: tf.data.Dataset

    # Number of top examples to display for each component.
    n_examples: int

    label_key: str
    example_keys: List[str]

    fontsize: str

    label_names: List[str]
    label_colors: Optional[List[str]] = None

    def __post_init__(self):
        assert len(self.label_names) == self.logits.shape[-1]
        if self.label_colors is not None:
            assert len(self.label_colors) == self.logits.shape[-1]

        assert self.W.shape[0] == self.logits.shape[0]

        self.n_npeff_examples, self.n_components = self.W.shape

        self.examples = self._make_examples()

        self._make_latex_label_defs()
        self._make_example_latex_defs()
        self._make_env_latex_defs()

    def _make_examples(self):
        return [
            Example(
                sentences=[tf.compat.as_str(x[k]) for k in self.example_keys],
                label=int(x[self.label_key])
            )
            for x in self.ds.as_numpy_iterator()
        ]

    def _make_latex_label_defs(self):
        max_chars = max(len(s) for s in self.label_names)
        padded_label_names = [s + ((max_chars - len(s)) * ' {}') for s in self.label_names]

        command_names = [f'\\LABEL{_commandify_name(s)}' for s in self.label_names]
        if len(command_names) != len(set(command_names)):
            raise ValueError('Failed to create unique command names for each label.')

        if self.label_colors is None:
            colors = len(command_names) * ['']
        else:
            colors = [R'\color{' + s + R'}' for s in self.label_colors]

        cmd_defs = [
            R'\newcommand{' + cmd + R'}{{' + color + R'\textbf{' + label + R'}}}'
            for label, cmd, color in zip(padded_label_names, command_names, colors)
        ]

        self.label_cmd_defs = '\n'.join(cmd_defs)
        self.label_to_label_cmd = command_names

    def _make_example_latex_defs(self):
        sub = [f'#{3 + i} ' + R'\vspace{0mm} \\' for i in range(len(self.example_keys))]
        s = _EXAMPLE_CMD_DEF_TEMPLATE.replace('@', f'{3 + len(self.example_keys)}')
        s = s.replace('$', '\n'.join(sub))

        self.example_cmd_def = s

    def _make_env_latex_defs(self):
        self.env_latex_def = _ENV_TEMPLATE.replace('@', self.fontsize)

    def _get_latex_defs(self) -> str:
        return '\n\n'.join([self.label_cmd_defs, self.example_cmd_def, self.env_latex_def])

    def _make_latex_for_example(self, example_index: int, component_index: int) -> str:
        example = self.examples[example_index]

        label = self.label_to_label_cmd[example.label]
        pred = self.label_to_label_cmd[np.argmax(self.logits[example_index])]
        coeff = f'{self.W[example_index, component_index]:.4f}'

        cmd1 = R'\compex{' + label + '}{' + pred + '}{' + coeff + '}'
        cmd2 = '\n'.join(['{' + latex_util.escape(s) + '}' for s in example.sentences])
        return '\n'.join([cmd1, cmd2])

    def _make_latex_for_component(self, component_index: int):
        top_inds = np.argsort(-self.W[:, component_index])[:self.n_examples]
        body = '\n'.join([self._make_latex_for_example(i, component_index) for i in top_inds])
        return '\n'.join([
            R'\begin{npeffcomp}' + f'[{component_index}]',
            body,
            R'\end{npeffcomp}',
        ])

    def make_latex(self, component_indices: Optional[List[int]] = None) -> str:
        if component_indices is None:
            component_indices = list(range(self.n_components))

        body = '\n\n'.join([
            self._make_latex_for_component(i)
            for i in component_indices
        ])

        return '\n\n'.join([
            _COMPONENTS_LATEX_FILE_START_TEMPLATE.replace('@', self._get_latex_defs()),
            body,
            _COMPONENTS_LATEX_FILE_END,
        ])
