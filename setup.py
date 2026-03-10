"""Setup configuration for scClone2DR package."""

from setuptools import find_namespace_packages, setup
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

setup(
    name="scClone2DR",
    version="0.1.0",
    author="Quentin Duchemin",
    author_email="qduchemin9@gmail.com",
    description="Clone-level multi-modal prediction of tumour drug response",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cbg-ethz/scClone2DR",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src", include=["scclone2dr*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.10.0",
        "pyro-ppl>=1.8.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "h5py>=3.0.0",
        "tqdm>=4.60.0",
        "scikit-learn>=0.24.0",
        "scikit-fda>=0.8.1",
        "nbformat>=5.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "flake8>=3.9",
            "isort>=5.9",
            "mypy>=0.910",
        ],
        "notebook": [
            "ipykernel",
            "jupyter",
            "nbformat>=5.0.0",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
