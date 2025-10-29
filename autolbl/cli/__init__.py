"""Command-line interface for autolbl."""

from autolbl.cli.prepare import main as prepare_main
from autolbl.cli.infer import main as infer_main

__all__ = ["prepare_main", "infer_main"]
