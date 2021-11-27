from functools import partial
from typing import Callable, NamedTuple

import chex
import jax
import optax


def ascent(rho, params, grads, eps=1e-6):
    """
    Updates parameters for a sharpness-aware ascent step as described in
    https://arxiv.org/abs/2010.01412.
    """
    nrm = jax.lax.max(optax.global_norm(grads), eps)
    inv = rho / nrm
    return jax.tree_multimap(lambda v, g: v + g * inv, params, grads)


def adaptive_ascent(rho, params, grads, eps=1e-6):
    """
    Adaptively updates parameters for a sharpness-aware ascent step as
    described in https://arxiv.org/abs/2102.11600.
    """
    ad_grad_norms = jax.tree_multimap(
        lambda g, v: (optax.safe_norm(g * jax.lax.abs(v), eps)), grads, params
    )
    ad_sam_params = jax.tree_multimap(
        lambda v, g, n: (v + jax.lax.square(v) * g * rho / n),
        params,
        grads,
        ad_grad_norms,
    )
    return ad_sam_params


class SAMState(NamedTuple):
    step: chex.Array
    g_v: optax.Params


def compute_g_v(g_s, g):
    nrm = optax.safe_norm(g, 1e-6)
    return -jax.lax.square(g) * g_s / jax.lax.square(nrm) + g_s


def make_sharpness_aware(
    inner: Callable[[optax.Params], optax.Updates],
    rho: float = 0.5,
    alpha: float = 1.0,
    look_freq: int = 5,
    adaptive: bool = True,
) -> optax.GradientTransformation:
    """
    Constructs a transform which wraps a forward pass to compute
    sharpness-aware gradients from a set of first-order updates given by the
    inner rule.

    inner:
        Callable which takes parameters as an argument and returns
        their gradients with respect to the desired objective.
    rho:
        Factor by which to scale the gradient norm during exact ascent steps.
    alpha:
        Additional factor by which to scale the gradient norm during approximate
        ascent steps.

    """

    assert 0 <= rho <= 1.0

    perturb = partial(adaptive_ascent if adaptive else ascent, rho)

    def init(params):
        return SAMState(step=0, g_v=jax.tree_map(jax.lax.zeros_like_array, params))

    def update(updates: optax.Updates, state: SAMState, params: optax.Params):
        assert (
            params is not None
        ), "params must be specified to perform the inner forward pass"

        g = jax.tree_map(partial(jax.lax.mul, -1), updates)

        def exact_update(params, g):
            g_s = inner(perturb(params, g))
            g_v = jax.tree_multimap(compute_g_v, g_s, g)
            return g_s, SAMState(step=look_freq, g_v=g_v), g_s

        def approx_update(params, g):
            del params
            g_v = state.g_v
            inv = optax.safe_norm(g) / optax.safe_norm(g_v)
            g_s = jax.tree_multimap(lambda g, g_v: g + alpha * inv * g_v, g, g_v)
            return g_s, SAMState(step=state.step - 1, g_v=g_v)

        return jax.lax.cond(state.step == 0, exact_update, approx_update, params, g)

    return optax.GradientTransformation(init, update)
