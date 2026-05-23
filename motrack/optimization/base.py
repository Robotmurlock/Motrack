"""
Common interface for HPO pipelines.

Every algorithm family (Optuna-driven samplers, MFGCS, …) implements
:class:`OptimizationPipeline`. The :func:`pipeline_factory` in
``factory.py`` returns instances of this type, so the dispatch site
in ``__init__.run_optimize`` is one line regardless of which algorithm
``cfg.optimizer.sampler`` selects.
"""
from abc import ABC, abstractmethod
from typing import Any

from motrack.config_parser import GlobalConfig
from motrack.tools.dataset_builder import DatasetBuilder


class OptimizationPipeline(ABC):
    """Run-interface for an HPO algorithm.

    Concrete subclasses receive the global config, a dataset builder, and a
    typed ``params`` payload (validated by the factory against a per-sampler
    dataclass). ``run`` performs the full optimization end-to-end and saves
    its results to disk via the standard ``conventions`` paths.
    """

    def __init__(
        self,
        cfg: GlobalConfig,
        dataset_builder: DatasetBuilder,
        params: Any,
    ) -> None:
        assert cfg.optimizer is not None, 'optimizer config is required'
        self._cfg = cfg
        self._dataset_builder = dataset_builder
        self._params = params

    @abstractmethod
    def run(self) -> None:
        """Execute the optimization end-to-end."""
