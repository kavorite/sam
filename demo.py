import itertools as it
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
        x = jax.random.truncated_normal(rng, lower=-3, upper=3, shape=(batch_size, 1))
        y = x * 3 + 1
        yield Batch(x, y)


def optimizer(steps: int, batch: Batch) -> optax.GradientTransformation:
    """
    Creates a sharpness-aware optimizer for a single step that closes over the
    given batch of input data.
    """

    def climb_fn(params: optax.Params) -> optax.Updates:
        # IMPORTANT: sharpness-aware ascent must close over the same data used
        # to perform the final gradient update
        return jax.grad(objective)(params, batch)

    sched = optax.linear_onecycle_schedule(steps, 0.1)
    optim = optax.inject_hyperparams(optax.sgd)(sched)

    return optax.chain(look_sharpness_aware(climb_fn), optim)


def train_init(steps: int, rng: jax.random.PRNGKey, batch: Batch) -> TrainState:
    "Initializes training state."
    params = model.init(rng, batch.x)
    opt_st = optimizer(steps, batch).init(params)
    return TrainState(params=params, opt_st=opt_st, loss=jnp.array(0.0), step=0)


def train_step(steps: int, state: TrainState, batch: Batch) -> TrainState:
    "Performs a single training step."
    loss, grads = jax.jit(jax.value_and_grad(objective))(state.params, batch)
    updates, opt_st = optimizer(steps, batch).update(grads, state.opt_st, state.params)
    params = optax.apply_updates(state.params, updates)
    step_inc = optax.safe_int32_increment(state.step)
    loss_avg = (state.loss * state.step + loss) / step_inc
    return TrainState(params=params, opt_st=opt_st, loss=loss_avg, step=step_inc)


def train(
    steps: int, batch_size: int, rng: jax.random.PRNGKey
) -> Generator[TrainState, None, None]:
    """
    Performs a sequence of training steps, yielding intermediate state to the caller.
    """
    data = dataset(rng, batch_size)
    batch = next(iter(data))
    state = train_init(steps, rng, batch)

    for batch in it.islice(data, steps):
        state = jax.jit(train_step, static_argnums=0)(steps, state, batch)
        yield state


try:
    steps = 1024
    batch_size = 32
    with tqdm(total=steps) as progress:
        for state in train(steps, batch_size, jax.random.PRNGKey(42)):
            progress.update()
            progress.set_description(f"{state.loss:.3g}")
except KeyboardInterrupt:
    pass
finally:
    if "state" in locals():
        print(state.params)
