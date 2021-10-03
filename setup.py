import os
from setuptools import find_packages, setup

setup(
    name="recurrent-intensity-model-experiments",
    version="1.0",
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},

    package_data={
        "rim_experiments": ["word_language_model/*.py"],
    },

    # use `conda env update --file environment.yml` to install full dependencies
    # "pynvml", # if on gpu host
    install_requires=[
        "torch>=1.7.1", # torch==1.7.1+cu101
        "pytorch-lightning>=1.3.8",
        "numba>=0.52.0",
        "lightfm>=1.16",
        "pyarrow>=0.13.0",
        "tick>=0.6",
        "implicit>=0.4.4",
        "backports.cached-property",
    ],
)
