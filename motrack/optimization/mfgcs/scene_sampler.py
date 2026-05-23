"""
Scene samplers for MFGCS low-fidelity evaluation.

A ``SceneSampler`` returns a small subset of scene names; the MFGCS algorithm
uses this subset to compare candidate parameter values cheaply before
validating the chosen value on the full dataset.
"""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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


@dataclass
class _StratifiedSceneSamplerParams:
    groups: Dict[str, str] = field(default_factory=dict)
    n_per_group: int = 2
    seed: int = 42
    strict: bool = True


class StratifiedSceneSampler(SceneSampler):
    """Sample ``n_per_group`` scenes from each regex-defined group.

    ``groups`` maps a group name to a regex pattern; each scene is
    assigned to the first group whose pattern matches it (via
    ``re.search``, so substrings work without anchors). The sampled
    subset is the concatenation of ``n_per_group`` scenes drawn
    uniformly without replacement from each group's bucket — yielding a
    balanced low-fidelity subset on multi-domain datasets like
    SPORTSMOT (basketball / football / volleyball).

    With ``strict=True`` (default), every scene must match exactly one
    group and every group must contribute at least one scene; otherwise
    a ``ValueError`` is raised. With ``strict=False``, non-matching
    scenes are dropped and empty groups are skipped silently.
    """

    def __init__(
        self,
        groups: Dict[str, str],
        n_per_group: int = 2,
        seed: int = 42,
        strict: bool = True,
    ) -> None:
        if n_per_group < 1:
            raise ValueError(f'n_per_group must be >= 1, got {n_per_group}')
        if not groups:
            raise ValueError('groups must contain at least one entry')
        self._groups: Dict[str, str] = dict(groups)
        self._compiled: Dict[str, 're.Pattern[str]'] = {
            name: re.compile(pattern) for name, pattern in self._groups.items()
        }
        self._n_per_group = n_per_group
        self._strict = strict
        self._rng = np.random.default_rng(seed)

    def sample(self, all_scenes: Sequence[str]) -> List[str]:
        scenes = list(all_scenes)
        if not scenes:
            raise ValueError('Cannot sample from an empty scene list')

        buckets: Dict[str, List[str]] = {g: [] for g in self._groups}
        unknown: List[str] = []
        for scene in scenes:
            for name, rx in self._compiled.items():
                if rx.search(scene):
                    buckets[name].append(scene)
                    break
            else:
                unknown.append(scene)

        if self._strict and unknown:
            raise ValueError(f'Scenes match no group regex: {sorted(unknown)}')

        out: List[str] = []
        for name in self._groups:
            pool = buckets[name]
            if not pool:
                if self._strict:
                    raise ValueError(f'Group {name!r} has zero scenes in all_scenes')
                continue
            k = min(self._n_per_group, len(pool))
            idx = self._rng.choice(len(pool), size=k, replace=False)
            out.extend(pool[int(i)] for i in idx)

        if not out:
            raise ValueError('Stratified sample produced an empty subset')
        return out


_SCENE_SAMPLER_REGISTRY: Dict[str, Any] = {
    'random': (RandomSceneSampler, _RandomSceneSamplerParams),
    'stratified': (StratifiedSceneSampler, _StratifiedSceneSamplerParams),
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
