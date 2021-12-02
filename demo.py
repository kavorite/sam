import itertools as it
from functools import partial
from typing import Generator, NamedTuple, Tuple

import chex
import haiku
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from sam import look_sharpness_aware

jax.config.update("jax_debug_nans", True)


class TrainState(NamedTuple):
    params: optax.Params
    opt_st: optax.OptState
    loss: float
    step: int


class Batch(NamedTuple):
    x: chex.Array
    y: chex.Array


@haiku.transform
def model(x: chex.Array) -> haiku.Module:
    "A trivial model for a trivial objective."
    return haiku.Linear(1)(x)


@jax.jit
def objective(params: optax.Params, batch: Batch) -> chex.Array:
    "A forward pass differentiable over model parameters."
    p = model.apply(params, None, batch.x)
    return jnp.square(p - batch.y).mean()


def dataset(rng: jax.random.PRNGKey, batch_size: int) -> Generator[Batch, None, None]:
    """
    A generator that yields a stream of synthetic batches from an
    algorithmic mock dataset.
    """
    while True:
        _, rng = jax.random.split(rng)
        x = jax.random.truncated_normal(rng, lower=-3, upper=3, shape=[batch_size, 1])
        y = x * 3 + 1
        yield Batch(x, y)


def optimizer(steps: int, batch: Batch) -> optax.GradientTransformation:
    """
    Creates a sharpness-aware optimizer for a single step that closes over the
    given batch of input data.
    """

    def lsched():
        return optax.linear_onecycle_schedule(steps, 0.1)

    def msched():
        sched = optax.linear_onecycle_schedule(steps, -0.1)

        def inner(step):
            return 1.0 - sched(step)

        return inner

    def forward(params: optax.Params) -> optax.Updates:
        # IMPORTANT: inner forward pass and sharpness-aware ascent must
        # close over the same data used to perform the forward pass in the final
        # gradient update
        return jax.tree_map(
            partial(jax.lax.mul, -1.0), jax.grad(objective)(params, batch)
        )

    inner = optax.inject_hyperparams(optax.adam)(learning_rate=lsched(), b1=msched())
    return look_sharpness_aware(inner, forward)


def train_init(steps: int, rng: jax.random.PRNGKey, batch: Batch) -> TrainState:
    "Initializes training state."
    params = model.init(rng, batch.x)
    opt_st = optimizer(steps, batch).init(params)
    return TrainState(params, opt_st, jnp.array(0.0), 0)


def cma_update(old: chex.Array, new: chex.Array, n: int) -> chex.Array:
    cma = n * old + new
    return cma / (n + 1)


def train_step(steps: int, state: TrainState, batch: Batch) -> TrainState:
    "Performs a single training step."
    loss, grads = jax.jit(jax.value_and_grad(objective))(state.params, batch)
    grads, optst = optimizer(steps, batch).update(grads, state.opt_st, state.params)
    params = optax.apply_updates(state.params, grads)
    loss = cma_update(state.loss, loss, state.step)
    return TrainState(params=params, opt_st=optst, loss=loss, step=state.step + 1)


def train(
    steps: int, batch_size: int, rng: jax.random.PRNGKey
) -> Generator[Tuple[chex.Scalar, TrainState], None, None]:
    """
    Performs a sequence of training steps, yielding intermediate state and
    loss to the caller.
    """
    data = dataset(rng, batch_size)
    batch = next(iter(data))
    state = train_init(steps, rng, batch)

    for batch in it.islice(data, steps):
        state = train_step(steps, state, batch)
        yield state


try:
    steps = 64
    batch_size = 32
    with tqdm(total=steps) as progress:

        for state in train(steps, batch_size, jax.random.PRNGKey(42)):
            progress.update()
            progress.set_description(f"{state.loss:.3g}")
except KeyboardInterrupt:
    pass
finally:
    print(state.params)
