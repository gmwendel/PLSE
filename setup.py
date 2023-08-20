# -*- coding: utf-8 -*-

"""
Installation script
"""

from __future__ import absolute_import

from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup, find_packages


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None

install_requires = [
    "matplotlib>=3.0",
    "scipy",
    "numpy>=1.19.2",
    "pandas",
    "seaborn"
]

setup(
    name="PLSE",
    description=(
        "A simple 1D CNN to count the number of photoelectrons in a waveform"
    ),
    author="Garrett Wendel et al.",
    author_email="gmw5164@psu.edu",
    url="tbd",
    license="Apache 2.0",
    version="0.0",
    python_requires=">=3.7",
    setup_requires=["tensorflow"],
    install_requires=install_requires,
    packages=find_packages(),
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'plse_train=plse.traincounter:main',
        ],
    },
)
