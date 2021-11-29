Implementation of [SAM][original] as a [jax]/[optax] `GradientTransformation`, with additional [adaptive] and [periodic] extensions. This codebase does not presently implement the layer-wise extensions specified in the latter report for large-batch training. See `demo.py` for a worked example of how to use the interface.

[original]: https://arxiv.org/abs/2010.01412
[periodic]: https://openreview.net/pdf?id=7VYh_3ZD84
[adaptive]: https://arxiv.org/abs/2102.11600
[optax]: https://github.com/deepmind/optax
[jax]: https://github.com/google/jax
