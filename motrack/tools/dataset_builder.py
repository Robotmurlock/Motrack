"""Pluggable dataset construction for inference / eval / optimize orchestration.

External libraries (e.g., custom dataset wrappers) can pass a custom
``DatasetBuilder`` to ``run_inference`` / ``run_eval`` / ``run_optimize``.
The default reproduces today's behavior: build via ``dataset_factory``.
"""
from typing import Callable

from motrack.config_parser import GlobalConfig
from motrack.datasets import BaseDataset, dataset_factory

DatasetBuilder = Callable[[GlobalConfig], BaseDataset]


def default_dataset_builder(cfg: GlobalConfig) -> BaseDataset:
    """Build dataset using ``motrack.datasets.dataset_factory`` (default behavior)."""
    return dataset_factory(
        dataset_type=cfg.dataset.type,
        path=cfg.dataset.fullpath,
        params=cfg.dataset.params,
        test=cfg.inference.split == 'test',
    )
