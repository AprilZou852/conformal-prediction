#!/usr/bin/env python3
"""Setup script for CPI (Conformalized Percentile Interval) package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="conformal-prediction",
    version="0.1.0",
    author="AprilZou852",
    description="Conformalized Percentile Interval: Finite Sample Validity and Improved Conditional Performance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AprilZou852/conformal-prediction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "torch>=1.10.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
        ],
    },
)
