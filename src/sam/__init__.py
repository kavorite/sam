from functools import partial
from typing import Callable, NamedTuple

import chex
import jax
import optax
from jax.tree_util import tree_map


def ascent(
    rho: float, params: optax.Params, grads: optax.Updates, eps: float = 1e-6
) -> optax.Params:
    """
    Updates parameters for a sharpness-aware ascent step as described in
    https://arxiv.org/abs/2010.01412.
    """
    nrm = jax.lax.max(optax.global_norm(grads), eps)
    inv = rho / nrm
    return tree_map(lambda v, g: v + g * inv, params, grads)


def adaptive_ascent(
    rho: float, params: optax.Params, grads: optax.Updates, eps: float = 1e-6
) -> optax.Params:
    """
    Adaptively updates parameters for a sharpness-aware ascent step as
    described in https://arxiv.org/abs/2102.11600.
    """
    ad_grad_norms = tree_map(
        lambda g, v: optax.safe_norm(g * jax.lax.abs(v), eps), grads, params
    )
    ad_sam_params = tree_map(
        lambda v, g, n: (v + jax.lax.square(v) * g * rho / n).astype(v.dtype),
        params,
        grads,
        ad_grad_norms,
    )
    return ad_sam_params


class AscentFn(Callable[[optax.Params, chex.ArrayTree], optax.Updates]):
    """
    Callable that closes a forward/backward pass over a given minibatch without
    fixing the parameters, returning raw gradients.
    """


def sharpness_aware(
    climb_fn: AscentFn,
    momentum: float = 0.05,
    adaptive: bool = False,
    eps: float = 1e-3,
) -> optax.GradientTransformation:
    """
    Constructs a transform which wraps a forward pass to compute
    sharpness-aware gradients, encouraging downstream gradient
    transforms to converge to smoother regions of their objective.

    forward:
        Callable that performs the same forward pass on the same data used to
        compute incoming gradients against the given set of parameters.
    rho:
        Hyperparameter controlling the magnitude of the ascent toward flatter
        regions of the loss landscape.
    adaptive:
        Whether to use adaptive scaling as in ASAM. Default momentum for this
        method is 2.0.
    """

    def init(params):
        del params
        return optax.EmptyState()

    def update(g, state, params):
        del state
        perturb = partial(adaptive_ascent if adaptive else ascent, momentum, eps=eps)
        g_s = climb_fn(perturb(params, g))
        return g_s, optax.EmptyState()

    return optax.GradientTransformation(init, update)


class LookSAState(NamedTuple):
    g_v: optax.Params
    skip: chex.Array  # scalar


def fast_g_v(g_s, g, eps=1e-6):
    nrm = optax.safe_norm(g, eps)
    g_s = g_s.astype(g.dtype)
    return -jax.lax.square(g) * g_s / jax.lax.square(nrm) + g_s


def look_sharpness_aware(
    climb_fn: AscentFn,
    rho: float = 0.05,
    adaptive: bool = True,
    skips: int = 5,
    scale: float = 1.0,
    eps: float = 1e-3,
) -> optax.GradientTransformation:
    """
    Variant of sharpness-aware optimization that only computes the true ascent
    vector every `skips` steps, and uses a projective approximation on
    intermediate steps.
    """
    inner = sharpness_aware(climb_fn, rho, adaptive, eps)

    def init(params):
        g_v = tree_map(jax.lax.zeros_like_array, params)
        return LookSAState(skip=0, g_v=g_v)

    def exact_update(g, state, params):
        del state
        g_s, empty = inner.update(g, optax.EmptyState(), params)
        del empty
        g_v = tree_map(partial(fast_g_v, eps=eps), g_s, g)
        return g_s, LookSAState(g_v=g_v, skip=skips)

    def apprx_update(g, state, params):
        del params
        g_v = state.g_v
        inv = optax.global_norm(g) / (optax.global_norm(g_v) + eps)
        g_s = tree_map(lambda g, g_v: g + scale * inv * g_v.astype(g.dtype), g, g_v)
        return g_s, LookSAState(g_v=g_v, skip=state.skip - 1)

    def update(g, state, params):
        return jax.lax.cond(
            state.skip == 0, exact_update, apprx_update, g, state, params
        )

    return optax.GradientTransformation(init, update)
