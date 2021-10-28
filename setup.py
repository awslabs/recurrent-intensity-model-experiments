import os
from setuptools import find_packages, setup

setup(
    name="recurrent-intensity-model-experiments",
    version="1.0",
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},

    # use `conda env update --file environment.yml` to install full dependencies
    install_requires=[
        "torch>=1.7.1", # torch==1.7.1+cu101
        "pytorch-lightning>=1.3.8",
        "numba>=0.52.0",
        "lightfm>=1.16",
        "pyarrow>=0.13.0",
        "tick>=0.6",
        # "implicit>=0.4.4", # conda install -c conda-forge implicit implicit-proc=*=gpu -y
        "backports.cached-property",
    ],
)
