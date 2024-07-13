"""
Installation script
"""

from setuptools import setup, find_packages

# List of required packages
install_requires = [
    "matplotlib>=3.0",
    "scipy",
    "numpy",
    "pandas",
    "seaborn",
    "tensorflow>=2.16"  # Default to CPU version of TensorFlow
]

# Optional dependencies
extras_require = {
    "gpu": ["tensorflow[and-cuda]>=2.16"],
}

# Setup configuration
setup(
    name="PLSE",
    description="A simple 1D CNN to count the number of photoelectrons in a waveform",
    author="Garrett Wendel et al.",
    author_email="gmw5164@psu.edu",
    url="tbd",
    license="Apache 2.0",
    version="0.0",
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    packages=find_packages(),
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'plse_train=plse.traincounter:main',
        ],
    },
)
