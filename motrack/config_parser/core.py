"""
Extension of core config that supports additional parameters for E2E evaluation.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore

from motrack.common import project
from motrack.utils.lookup import LookupTable


@dataclass
class PathConfig:
    """
    Path configs
    - master: location where all final and intermediate results are stored
    - assets: location where datasets can be found
    """
    master: str = project.MASTER_PATH
    assets: str = project.ASSETS_PATH

    @classmethod
    def default(cls) -> 'PathConfig':
        """
        Default path configuration is used if it is not defined in configs.

        Returns: Path configuration.
        """
        return cls(
            master=project.OUTPUTS_PATH,
            assets=project.ASSETS_PATH
        )


@dataclass
class DatasetConfig:
    """
    Dataset config.
    - name: Dataset name
    - path: Path to the dataset
    """
    type: str
    path: str
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Fullpath is resolved from global config (path is required)
        """
        self.fullpath = None


@dataclass
class FilterConfig:
    type: str
    params: dict


@dataclass
class ObjectDetectionInferenceConfig:
    type: str
    params: dict
    lookup_path: Optional[str] = None
    cache_path: Optional[str] = None
    oracle: bool = False

    def load_lookup(self) -> LookupTable:
        """
        Loads lookup file.

        Returns:
            Loaded lookup
        """
        with open(self.lookup_path, 'r', encoding='utf-8') as f:
            return LookupTable.deserialize(json.load(f))


@dataclass
class TrackerAlgorithmConfig:
    name: str
    params: dict


@dataclass
class TrackerVisualizeConfig:
    fps: int = 20
    new_object_length: int = 5
    option: str = 'active'

    def __post_init__(self) -> None:
        """
        Validation.
        """
        options = ['active', 'all', 'postprocess']
        assert self.option in options, f'Invalid option "{self.option}". Available: {options}.'


@dataclass
class TrackerPostprocessConfig:
    init_threshold: int = 2  # Activate `init_threshold` starting bboxes
    linear_interpolation_threshold: int = 3  # Maximum distance to perform linear interpolation
    min_tracklet_length: int = 20  # Remove all tracklets that are shorter than this


@dataclass
class TrackerEvalConfig:
    split: str


@dataclass
class FilterConfig:
    scene_pattern: str = '(.*?)'  # All


@dataclass
class GlobalConfig:
    experiment: str
    dataset: DatasetConfig
    eval: TrackerEvalConfig
    object_detection: ObjectDetectionInferenceConfig
    algorithm: TrackerAlgorithmConfig
    filter: FilterConfig = field(default_factory=FilterConfig)
    path: PathConfig = field(default_factory=PathConfig)
    postprocess: TrackerPostprocessConfig = field(default_factory=TrackerPostprocessConfig)
    visualize: TrackerVisualizeConfig = field(default_factory=TrackerVisualizeConfig)

    @property
    def experiment_path(self) -> str:
        """
        Path where tracker results can be found.
        """
        return os.path.join(self.path.master, self.dataset.type, self.algorithm.name, self.experiment, self.eval.split)

    def __post_init__(self):
        """
        Postprocess.
        """
        self.dataset.fullpath = os.path.join(self.path.assets, self.dataset.path, self.eval.split)


# Configuring hydra config store
# If config has `- global_config` in defaults then
# full config is recursively instantiated
cs = ConfigStore.instance()
cs.store(name='global_config', node=GlobalConfig)
