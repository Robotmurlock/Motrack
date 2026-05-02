"""
Sampler-name → :class:`OptimizationPipeline` factory.

Every algorithm (``random`` / ``tpe`` / ``warm_tpe`` / ``mfgcs``) registers
itself with a typed params dataclass. ``sampler_params`` from the YAML is
validated against that dataclass at construction time, so typos surface as
``TypeError`` instead of silently no-opping.

Mirrors the existing ``scene_sampler_factory`` / ``coordinate_optimizer_factory``
shape so the project has a single factory pattern.
"""
from typing import Dict, Tuple, Type

from motrack.config_parser import GlobalConfig
from motrack.tools.dataset_builder import DatasetBuilder
from motrack.optimization.base import OptimizationPipeline


_REGISTRY: Dict[str, Tuple[Type[OptimizationPipeline], Type]] = {}


def register_pipeline(
    name: str,
    pipeline_cls: Type[OptimizationPipeline],
    params_cls: Type,
) -> None:
    """Register a pipeline implementation under ``name``."""
    if name in _REGISTRY:
        raise ValueError(f'Pipeline already registered for sampler "{name}".')
    _REGISTRY[name] = (pipeline_cls, params_cls)


def pipeline_factory(
    sampler: str,
    sampler_params: dict,
    cfg: GlobalConfig,
    dataset_builder: DatasetBuilder,
) -> OptimizationPipeline:
    """Construct the pipeline for ``sampler`` with validated ``sampler_params``."""
    if sampler not in _REGISTRY:
        raise ValueError(
            f'Unknown sampler: "{sampler}". Registered: {sorted(_REGISTRY)}.'
        )
    pipeline_cls, params_cls = _REGISTRY[sampler]
    typed_params = params_cls(**(sampler_params or {}))
    return pipeline_cls(cfg, dataset_builder, params=typed_params)
