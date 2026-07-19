"""Real-world graph signals and reproducible benchmarks."""

from .datasets import available_datasets, load_dataset
from .models import GraphSignalDataset

__all__ = ["GraphSignalDataset", "available_datasets", "load_dataset"]
__version__ = "0.0.1"
