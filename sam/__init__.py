from functools import partial
from typing import Callable, NamedTuple

import chex
import jax
import optax


def ascent(
    rho: float, params: optax.Params, grads: optax.Updates, eps: float = 1e-6
) -> optax.Params:
    """
    Updates parameters for a sharpness-aware ascent step as described in
    https://arxiv.org/abs/2010.01412.
    """
    nrm = jax.lax.max(optax.global_norm(grads), eps)
    inv = rho / nrm
    return jax.tree_multimap(lambda v, g: v + g * inv, params, grads)


def adaptive_ascent(
    rho: float, params: optax.Params, grads: optax.Updates, eps: float = 1e-6
) -> optax.Params:
    """
    Adaptively updates parameters for a sharpness-aware ascent step as
    described in https://arxiv.org/abs/2102.11600.
    """
    ad_grad_norms = jax.tree_multimap(
        lambda g, v: optax.safe_norm(g * jax.lax.abs(v), eps), grads, params
    )
    ad_sam_params = jax.tree_multimap(
        lambda v, g, n: (v + jax.lax.square(v) * g * rho / n).astype(v.dtype),
        params,
        grads,
        ad_grad_norms,
    )
    return ad_sam_params


class ForwardFn(Callable[[optax.Params, chex.ArrayTree], optax.Updates]):
    """
    Callable that performs the forward pass of a user-defined objective on the
    same batch with new parameters during sharpness-aware optimization.
    """


def sharpness_aware(
    forward: ForwardFn, rho: float = 0.5, adaptive: bool = True, eps: float = 1e-3
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
        Whether to use adaptive scaling as in ASAM.
    """

    def init(params):
        del params
        return optax.EmptyState()

    def update(g, state, params):
        del state
        perturb = partial(adaptive_ascent if adaptive else ascent, rho, eps=eps)
        g_s = forward(perturb(params, g))
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
    forward: ForwardFn,
    rho: float = 0.5,
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
    inner = sharpness_aware(forward, rho, adaptive, eps)

    def init(params):
        g_v = jax.tree_map(jax.lax.zeros_like_array, params)
        return LookSAState(skip=0, g_v=g_v)

    def exact_update(g, state, params):
        del state
        g_s, empty = inner.update(g, optax.EmptyState(), params)
        del empty
        g_v = jax.tree_multimap(partial(fast_g_v, eps=eps), g_s, g)
        return g_s, LookSAState(g_v=g_v, skip=skips)

    def apprx_update(g, state, params):
        del params
        g_v = state.g_v
        inv = optax.global_norm(g) / (optax.global_norm(g_v) + eps)
        g_s = jax.tree_multimap(
            lambda g, g_v: g + scale * inv * g_v.astype(g.dtype), g, g_v
        )
        return g_s, LookSAState(g_v=g_v, skip=state.skip - 1)

    def update(g, state, params):
        return jax.lax.cond(
            state.skip == 0, exact_update, apprx_update, g, state, params
        )

    return optax.GradientTransformation(init, update)
