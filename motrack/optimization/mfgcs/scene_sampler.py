"""
Scene samplers for MFGCS low-fidelity evaluation.

A ``SceneSampler`` returns a small subset of scene names; the MFGCS algorithm
uses this subset to compare candidate parameter values cheaply before
validating the chosen value on the full dataset.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from motrack.config_parser import FactorySpec


class SceneSampler(ABC):
    """Pick a small subset of scenes for low-fidelity evaluation."""

    @abstractmethod
    def sample(self, all_scenes: Sequence[str]) -> List[str]:
        """Return the sampled scene names. Must be a subset of ``all_scenes``."""


@dataclass
class _RandomSceneSamplerParams:
    n: int = 8
    seed: int = 42


class RandomSceneSampler(SceneSampler):
    """Sample ``n`` scenes uniformly at random.

    A new random subset is drawn on every ``sample()`` call (the algorithm
    advances the RNG state per coordinate sweep, satisfying the "same
    sampled scenes inside the optimizer, fresh sample per coordinate" rule
    from the algorithm spec).
    """

    def __init__(self, n: int = 8, seed: int = 42) -> None:
        if n < 1:
            raise ValueError(f'n must be >= 1, got {n}')
        self._n = n
        self._rng = np.random.default_rng(seed)

    def sample(self, all_scenes: Sequence[str]) -> List[str]:
        scenes = list(all_scenes)
        if not scenes:
            raise ValueError('Cannot sample from an empty scene list')
        n = min(self._n, len(scenes))
        idx = self._rng.choice(len(scenes), size=n, replace=False)
        return [scenes[int(i)] for i in idx]


_SCENE_SAMPLER_REGISTRY: Dict[str, Any] = {
    'random': (RandomSceneSampler, _RandomSceneSamplerParams),
}


def scene_sampler_factory(spec: FactorySpec) -> SceneSampler:
    """Construct a scene sampler from a ``FactorySpec`` declaration.

    Raises:
        ValueError: if ``spec.type`` is not registered.
        TypeError: if ``spec.params`` contains keys the variant doesn't accept.
    """
    if spec.type not in _SCENE_SAMPLER_REGISTRY:
        raise ValueError(
            f'Unknown scene sampler type: "{spec.type}". '
            f'Known: {sorted(_SCENE_SAMPLER_REGISTRY)}'
        )
    cls, params_cls = _SCENE_SAMPLER_REGISTRY[spec.type]
    typed = params_cls(**(spec.params or {}))
    return cls(**typed.__dict__)
