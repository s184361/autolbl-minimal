"""Setup configuration for AutoLbl package"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="autolbl",
    version="0.1.0",
    author="Jędrzej Kolbert",
    author_email="s184361@dtu.dk",
    description="Automatic Labelling for Image Datasets using Vision-Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/s184361/autolbl",
    packages=find_packages(include=["autolbl", "autolbl.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11,<3.13",
    install_requires=[
        # Core dependencies from pyproject.toml
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.21.2",
        "opencv-python>=4.10.0",
        "pillow>=11.0.0",
        "supervision>=0.20.0",
        "pandas>=2.2.0",
        "autodistill>=0.1.29",
        "wandb",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "autolbl-prepare=scripts.prepare_datasets:main",
            "autolbl-infer=scripts.run_inference:main",
        ],
    },
)
