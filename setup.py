from setuptools import find_namespace_packages, setup

setup(
    name="sam",
    version="1.0.1",
    description="Sharpness-aware minimization in optax",
    author="kavorite",
    url="https://github.com/kavorite/sam",
    install_requires=["dm_haiku>=0.0.5" "jax>=0.2.27" "optax>=0.0.9"],
    packages=find_namespace_packages(),
)
