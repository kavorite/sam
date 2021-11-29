import itertools as it

import haiku
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from sam import look_sharpness_aware

STEPS = 256


def model(x):
    return haiku.Linear(1)(x)


def objective(params, x, y):
    p = haiku.transform(model).apply(params, None, x)
    return jnp.square(p - y).mean()


def forward(params, opt_st, x, y):
    loss, grads = jax.value_and_grad(objective)(params, x, y)
    return loss, *optimizer().update(grads, opt_st)


def dataset(rng):
    while True:
        _, rng = jax.random.split(rng)
        x = jax.random.truncated_normal(rng, -3, 3)
        y = x * 3 + 1
        yield jnp.array([x]), jnp.array([y])


rng = jax.random.PRNGKey(42)
data = dataset(rng)
x, _ = next(iter(data))
params = haiku.transform(model).init(rng, x)


def cma_update(old, new, n):
    cma = n * old + new
    return cma / (n + 1)


try:
    with tqdm(total=STEPS) as progress:
        current_batch = None
        cum_loss = 0.0

        def optimizer():
            def forward(params):
                return jax.grad(objective)(params, *current_batch)

            return optax.chain(
                look_sharpness_aware(forward),
                optax.inject_hyperparams(optax.sgd)(
                    learning_rate=optax.linear_onecycle_schedule(STEPS, 0.1)
                ),
            )

        optim = optimizer()
        optst = optim.init(params)

        for i, (x, y) in enumerate(it.islice(data, STEPS)):
            current_batch = (x, y)
            loss, grads = jax.jit(jax.value_and_grad(objective))(params, x, y)
            cum_loss = cma_update(cum_loss, loss, i + 1)
            if jnp.isnan(loss):
                break
            grads, optst = optim.update(grads, optst, params)
            params = optax.apply_updates(params, grads)
            progress.update()
            progress.set_description(f"{cum_loss:.3g}")
except KeyboardInterrupt:
    pass
finally:
    print(params)
