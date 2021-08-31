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

    install_requires=[ # check out environment.yml for full list
        "torch>=1.7.1", # torch==1.7.1+cu101
        "pytorch-lightning>=1.3.8",
        "numba>=0.52.0",
        # "tick>=0.6",
        "lightfm>=1.16",
        "pyarrow>=0.13.0",
    ],
)
