"""Perturbation utilties for LRM-NPEFF."""
import dataclasses
from typing import List, Union

import numpy as np
import tensorflow as tf

from npeff.decomp import decomps
from npeff.util import flat_pack

from . import evaluation


@dataclasses.dataclass
class LrmPerturbationHelper:
    decomp: decomps.LrmNpeffDecomposition

    model: tf.keras.Model
    variables: List[tf.Variable]

    eval_ctx: evaluation.EvaluationContext

    n_top_examples: int
    n_baseline_examples: int

    def __post_init__(self):
        self.n_examples = self.decomp.W.shape[0]

        self.og_variables = [tf.identity(v) for v in self.variables]

        # Normalize the components here just to be safe.
        self.decomp.normalize_components_to_unit_norm()

        # Precompute this for faster per-component runs downs the line.
        self.abs_cos_sims = np.abs(self.decomp.G @ self.decomp.G.T)

###############################################################################


@dataclasses.dataclass
class PerturbationResults:
    # Results of evaluation on the top examples for a component.
    top_results: evaluation.EvaluationResults
    # Results of evaluation on a baseline set of examples.
    baseline_results: evaluation.EvaluationResults
    
    def kl_ratio(self):
        return self.top_results.kl() / self.baseline_results.kl()


@dataclasses.dataclass
class PmPerturbationResults:
    # L2 norm of the perturbation.
    magnitude: float
    # Results of perturbation in the direction of +g.
    plus_results: PerturbationResults
    # Results of perturbation in the direction of -g.
    minus_results: PerturbationResults

    def max_kl_ratio(self):
        return max(self.plus_results.kl_ratio(), self.minus_results.kl_ratio())

    def max_top_examples_acc(self):
        return max(self.plus_results.top_results.acc(), self.minus_results.top_results.acc())

    def max_baseline_examples_acc(self):
        return max(self.plus_results.baseline_results.acc(), self.minus_results.baseline_results.acc())


@dataclasses.dataclass
class LrmComponentPerturber:
    helper: LrmPerturbationHelper

    component_index: int

    # Set to None for no semi-orthogonalization.
    max_sim: Union[float, None]

    def __post_init__(self):
        if self.max_sim is not None:
            assert 0.0 < self.max_sim < 1.0

        self.decomp = self.helper.decomp
        self.abs_cos_sims = self.helper.abs_cos_sims

        self.model = self.helper.model
        self.variables = self.helper.variables
        self.og_variables = self.helper.og_variables

        self.eval_ctx = self.helper.eval_ctx

        self.packer = flat_pack.FlatPacker([v.shape for v in self.variables])

        self.top_inds = np.argsort(-self.decomp.W[:, self.component_index])[:self.helper.n_top_examples]
        self._baseline_inds = np.arange(self.helper.n_baseline_examples)

        self._normalized_perturbation = None

    @property
    def normalized_perturbation(self) -> List[tf.Tensor]:
        if self._normalized_perturbation is None:
            # Assumes rows of G have unit norm.
            G = self.decomp.G

            g_main = np.copy(G[self.component_index])

            if self.max_sim is not None:
                for i in range(G.shape[0]):
                    if i == self.component_index:
                        continue
                    if self.abs_cos_sims[self.component_index, i] > self.max_sim:
                        continue
                    g_main -= g_main.dot(G[i]) * G[i]

            g_main /= np.sqrt(np.sum(g_main**2))

            g = np.zeros([self.decomp.n_parameters], dtype=np.float32)
            g[self.decomp.new_to_old_col_indices] = g_main

            self._normalized_perturbation = self.packer.decode_tf(tf.cast(g, tf.float32))

        return self._normalized_perturbation

    def _perturb_weights(self, multiplier: float):
        for ogv, v, offset in zip(self.og_variables, self.variables, self.normalized_perturbation):
            v.assign(ogv + multiplier * offset)

    def _evaluate_model(self) -> PerturbationResults:
        top_results = self.eval_ctx.evaluate(self.model, self.top_inds)
        baseline_results = self.eval_ctx.evaluate(self.model, self._baseline_inds)
        return PerturbationResults(top_results=top_results, baseline_results=baseline_results)

    def evaluate_pm(self, magnitude: float) -> PmPerturbationResults:
        """Evaluates a perturbation for both +/- magnitude * normalized_perturbation."""
        assert magnitude >= 0.0

        self._perturb_weights(magnitude)
        plus_results = self._evaluate_model()

        self._perturb_weights(-magnitude)
        minus_results = self._evaluate_model()

        return PmPerturbationResults(
            magnitude=magnitude,
            plus_results=plus_results,
            minus_results=minus_results,
        )
