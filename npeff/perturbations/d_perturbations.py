"""Perturbation experiments for D-NPEFF decompositions."""
import dataclasses
import random
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

from npefs.models import npeff_models
from npeff.decomp import decomps
from npeff.fishers import dsfs
from npeff.util import flat_pack

from . import evaluation


# kl range and targeter params
# dsf fisher (need to add code to compute it)
# sign pattern stuff: I think just compute gradients and similar


##########################################################################


class InvalidKlFnOutputError(Exception):
    pass


@dataclasses.dataclass
class KlTargeter:
    kl_range: Tuple[float, float]
    delta_mag_range: Tuple[float, float]
    lmbda_working_range: Tuple[float, float] = (0.035, 1 - 0.035)

    max_iters: int

    backoff_factor: float = 2.0
    backoff_attempts: int = 10

    def __post_init__(self):
        assert len(self.kl_range) == 2
        assert 0 < self.min_kl < self.max_kl

        assert len(self.delta_mag_range) == 2
        assert 0 < self.min_delta_mag < self.max_delta_mag

        assert len(self.lmbda_working_range) == 2
        assert 0 < self.min_lmbda < self.max_lmbda < 1

        assert self.backoff_factor > 1

        self.kl_fn = None

        self._last_delta = None
        self._last_lmbda = None

    def sorta_clone(self):
        # Won't copy over the kl_fn or anything else.
        return self.__class__(**dataclasses.asdict(self))

    def set_kl_fn(self, kl_fn: Callable[[float, float], float]):
        # kl_fn: [delta, lmbda] -> kl
        self.kl_fn = kl_fn

    #################################################################

    @property
    def min_kl(self):
        return self.kl_range[0]

    @property
    def max_kl(self):
        return self.kl_range[1]

    @property
    def min_delta_mag(self):
        return self.delta_mag_range[0]

    @property
    def max_delta_mag(self):
        return self.delta_mag_range[1]

    @property
    def min_lmbda(self):
        return self.lmbda_working_range[0]

    @property
    def max_lmbda(self):
        return self.lmbda_working_range[1]

    #################################################################

    def _get_random_delta(self):
        # Log uniform distribution.
        log_delta = random.uniform(np.log(self.min_delta_mag), np.log(self.max_delta_mag))
        return np.exp(log_delta)

    def _get_random_lmbda(self):
        # Uniform distribution
        return random.uniform(self.min_lmbda, self.max_lmbda)

    #################################################################

    def _evaluate(self, delta, lmbda):
        self._last_delta = delta
        self._last_lmbda = lmbda

        kl = self.kl_fn(delta, lmbda)

        if self.min_kl <= kl <= self.max_kl:
            return (0, kl)
        elif kl < self.min_kl:
            return (-1, kl)
        elif self.max_kl < kl:
            return (1, kl)
        else:
            raise InvalidKlFnOutputError('This condition should not be reachable.')

    def _kl_step_coeffs_gen(self, delta, lmbda, condition, i):
        assert condition != 0

        # if random.random() < 0.5:
        if i % 2:
            # Do delta.
            og_log_delta = np.log(delta)

            if condition < 0:
                log_diff = (np.log(self.max_delta_mag) - og_log_delta) / 2
            else:
                log_diff = (np.log(self.min_delta_mag) - og_log_delta) / 2

            for i in range(self.backoff_attempts):
                new_delta = np.exp(og_log_delta + log_diff)
                yield new_delta, lmbda
                log_diff /= self.backoff_factor

        else:
            # Do lmbda
            og_lmbda = lmbda

            if condition < 0:
                diff = (self.max_lmbda - og_lmbda) / 2
            else:
                diff = (self.min_lmbda - og_lmbda) / 2

            for i in range(self.backoff_attempts):
                new_lmbda = og_lmbda + diff
                yield delta, new_lmbda
                diff /= self.backoff_factor

    def search(
        self,
        init_delta: Optional[float] = None,
        init_lmbda: Optional[float] = None,
    ):
        # NOTE: This is based on the assumption that the KL is monotonic in both
        # delta and lmbda.
        # ablating_fisher = self._get_ablating_fisher(component_index)
        delta = init_delta or self._get_random_delta()
        lmbda = init_lmbda or self._get_random_lmbda()

        for i in range(self.max_iters):
            condition, kl0 = self._evaluate(delta, lmbda)
            if condition == 0:
                return True

            coeffs = list(self._kl_step_coeffs_gen(delta, lmbda, condition, i))
            for j, (delta, lmbda) in enumerate(coeffs):
                cond, _ = self._evaluate(delta, lmbda)
                if cond == 0:
                    return True
                elif cond == condition:
                    break

        return False


##########################################################################


@dataclasses.dataclass
class SignPatternGenerator:

    model: tf.keras.Model
    variables: List[tf.Variable]

    eval_ctx: evaluation.EvaluationContext

    batch_size: int = 8

    def __post_init__(self):
        # Sign of gradient of loss over set of examples.
        self.sign_pattern = [tf.Variable(tf.zeros_like(v), trainable=False) for v in self.variables]

    ############################################################
    # I think needed to be hashable to work with tf.function
    
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    ############################################################

    @tf.function
    def _zero_out_sign_pattern(self):
        for g in self.sign_pattern:
            g.assign(tf.zeros_like(g))

    @tf.function
    def _make_sign_pattern_from_gradient(self):
        for g in self.sign_pattern:
            g.assign(tf.sign(g))

    @tf.function
    def _update_sign_pattern_with_gradient(self, x, y):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.variables)
            logits = npeff_models.compute_logits(self.models, x, training=False)
            loss = self.model.compiled_loss(y, logits)

        grads = tape.gradient(loss, self.variables)
        for f, g in zip(self.sign_pattern, grads):
            f.assign_add(g)

    ############################################################

    def update_sign_pattern(self, example_indices: np.ndarray) -> List[tf.Tensor]:
        ds = self.eval_ctx.get_ds(example_indices)
        ds = ds.batch(self.batch_size)

        self._zero_out_sign_pattern()

        for x, y in ds:
            self._update_sign_pattern_with_gradient(x, y)

        self._make_sign_pattern_from_gradient()


###############################################################################


@dataclasses.dataclass
class DPerturbationHelper:
    decomp: Union[decomps.DNpeffDecomposition, decomps.SparseDNpeffDecomposition]

    d_dsf: dsfs.DenseDDsf

    model: tf.keras.Model
    variables: List[tf.Variable]

    eval_ctx: evaluation.EvaluationContext

    kl_targeter: KlTargeter

    n_top_examples: int
    n_baseline_examples: int

    fisher_floor: float = 1e-6,

    sign_pattern_batch_size: int = 8

    def __post_init__(self):
        self.n_examples = self.decomp.W.shape[0]

        self.og_variables = [tf.identity(v) for v in self.variables]

        self.packer = flat_pack.FlatPacker([v.shape for v in self.variables])

        self.sign_pattern_generator = SignPatternGenerator(
            model=self.model,
            variables=self.variables,
            eval_ctx=self.eval_ctx,
            batch_size=self.sign_pattern_batch_size,
        )

        # Normalize here just to be safe.
        self.d_dsf.normalize_to_unit_norm()

        self.dataset_fisher = self.packer.decode_tf(tf.cast(self.d_dsf.fisher, tf.float32))

    def make_component_perturber(self, component_index: int) -> 'DComponentPerturber':
        # NOTE: This shares some state with the helper and objects owned by it. So
        # only one of the component perturbers should be used at a time.
        return DComponentPerturber(
            component_index=component_index,
            helper=self,
            kl_targeter=self.kl_targeter.sorta_clone(),
        )


@dataclasses.dataclass
class PerturbationResults:
    # Results of evaluation on the top examples for a component.
    top_results: evaluation.EvaluationResults
    # Results of evaluation on a baseline set of examples.
    baseline_results: evaluation.EvaluationResults
    
    def kl_ratio(self):
        return self.top_results.kl() / self.baseline_results.kl()


@dataclasses.dataclass
class DComponentPerturber:
    helper: DPerturbationHelper

    component_index: int

    kl_targeter: KlTargeter

    def __post_init__(self):
        self.model = self.helper.model
        self.eval_ctx = self.helper.eval_ctx

        self.variables = self.helper.variables
        self.og_variables = self.helper.og_variables
        self.dataset_fisher = self.helper.dataset_fisher
        self.sign_pattern = self.helper.sign_pattern_generator.sign_pattern

        self.fisher_floor = self.helper.fisher_floor

        self.top_inds = np.argsort(-self.helper.decomp.W[:, self.component_index])[:self.helper.n_top_examples]
        self.baseline_inds = np.arange(self.helper.n_baseline_examples)

        self.kl_targeter.set_kl_fn(self._kl_fn)

        # This is normalized to unit L2 norm.
        self.comp_fisher = self._make_comp_fisher()

        self.helper.sign_pattern_generator.update_sign_pattern(self.top_inds)

    def _make_comp_fisher(self):
        h = self.helper.decomp.get_h_vector(self.component_index)
        h /= np.sqrt(np.sum(h**2))
        return self.packer.decode_tf(tf.cast(h, tf.float32))

    ############################################################
    # I think needed to be hashable to work with tf.function
    
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    ############################################################

    def _set_model_params(self, delta: float, lmbda: float):
        # Need to do it this way to prevent re-tracing with every call.
        self._set_model_params_tf(tf.cast(delta, tf.float32), tf.cast(lmbda, tf.float32))

    @tf.function
    def _set_model_params_tf(self, delta, lmbda):
        zipped = zip(self.variables, self.og_variables, self.sign_pattern, self.dataset_fisher, self.comp_fisher)
        for v, og_v, sp, df, cf in zipped:
            df = tf.minimum(df, self.fisher_floor)

            p_v = og_v - delta * sp

            w1 = (1 - lmbda) * df
            w2 = lmbda * cf

            v.assign((w1 * og_v + w2 * p_v) / (w1 + w2))

    def _kl_fn(self, delta: float, lmbda: float):
        self._set_model_params(delta, lmbda)
        return self.eval_ctx.evaluate(self.model, self.top_inds).kl()

    def evaluate_perturbation(self) -> Union[PerturbationResults, None]:
        found = self.kl_targeter.search()

        if not found:
            return None

        top_results = self.eval_ctx.evaluate(self.model, self.top_inds)
        baseline_results = self.eval_ctx.evaluate(self.model, self.baseline_inds)

        return PerturbationResults(top_results=top_results, baseline_results=baseline_results)
