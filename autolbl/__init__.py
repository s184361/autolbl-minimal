"""
AutoLbl: Automatic Labelling for Image Datasets

A framework for automatic image annotation using vision-language models (VLMs).
"""

__version__ = "0.1.0"

from . import models
from . import data
from . import evaluation
from . import ontology
from . import visualization

__all__ = ["models", "data", "evaluation", "ontology", "visualization"]
