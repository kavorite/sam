import itertools as it

import haiku
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from sam import look_sharpness_aware

STEPS = 256

# Create the model
def model(x):
    return haiku.Linear(1)(x)


# Define a forward pass differentiable over model parameters
def objective(params, x, y):
    p = haiku.transform(model).apply(params, None, x)
    return jnp.square(p - y).mean()


# Synthetic, linearly-separable toy data
def dataset(rng):
    while True:
        _, rng = jax.random.split(rng)
        x = jax.random.truncated_normal(rng, -3, 3)
        y = x * 3 + 1
        yield jnp.array([x]), jnp.array([y])


# prepare the dataset
rng = jax.random.PRNGKey(42)
data = dataset(rng)
x, _ = next(iter(data))
params = haiku.transform(model).init(rng, x)


# cumulative moving average to monitor metrics over time
def cma_update(old, new, n):
    cma = n * old + new
    return cma / (n + 1)


jax.config.update("jax_debug_nans", True)
try:
    with tqdm(total=STEPS) as progress:
        current_batch = None
        cum_loss = 0.0

        def optimizer():
            def forward(params):
                # IMPORTANT: inner forward pass and sharpness-aware ascent must
                # close over the same data used to perform the final gradient
                # update
                return jax.grad(objective)(params, *current_batch)

            return optax.chain(
                look_sharpness_aware(forward),
            )

        optim = optimizer()
        optst = optim.init(params)

        for i, (x, y) in enumerate(it.islice(data, STEPS)):
            current_batch = (x, y)
            loss, grads = jax.jit(jax.value_and_grad(objective))(params, x, y)
            cum_loss = cma_update(cum_loss, loss, i + 1)
            grads, optst = optim.update(grads, optst, params)
            params = optax.apply_updates(params, grads)
            progress.update()
            progress.set_description(f"{cum_loss:.3g}")
except KeyboardInterrupt:
    pass
finally:
    print(params)
