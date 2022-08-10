from setuptools import find_namespace_packages, setup

setup(
    name="sam",
    version="0.0.2",
    description="Sharpness-aware minimization in optax",
    author="kavorite",
    url="https://github.com/kavorite/sam",
    install_requires=["dm_haiku>=0.0.5" "jax>=0.2" "optax>=0.1"],
    packages=find_namespace_packages(),
)
