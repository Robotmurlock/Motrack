"""
Tracker config for tool entrypoints.
"""
from datetime import datetime
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore

from motrack.common import conventions, formats, project
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
    - name: Optional dataset output name
    - path: Path to the dataset
    - params: Ctor parameters
    """
    type: str
    path: str
    name: Optional[str] = None
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Fullpath is resolved from global config (path is required)
        """
        self.fullpath = None
        self.basepath = None


@dataclass
class FilterConfig:
    """
    Filter (motion model) config.
    """
    type: str
    params: dict


@dataclass
class ObjectDetectionInferenceConfig:
    """
    Object detection config.
    - type: Architecture
    - params: Ctor params
    - lookup_path: Class mappings saved as a json file (in LookupTable format)
    - cache_path: Use cache for faster inference
    - oracle: Use GT detections instead running inference
    """
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
    requires_image: bool = False


@dataclass
class TrackerVisualizeConfig:
    fps: int = 20
    new_object_length: int = 5
    option: str = 'debug'
    is_rgb: bool = field(default=True)

    def __post_init__(self) -> None:
        """
        Validation.
        """
        self.option = conventions.normalize_tracker_output_name(self.option)


@dataclass
class TrackerPostprocessConfig:
    init_threshold: int = 2  # Activate `init_threshold` starting bboxes
    linear_interpolation_threshold: int = 3  # Maximum distance to perform linear interpolation
    linear_interpolation_min_tracklet_length: int = 30  # Minimum tracklet length to perform linear interpolation
    min_tracklet_length: int = 20  # Remove all tracklets that are shorter than this


@dataclass
class TrackerEvalConfig:
    split: str
    postprocess: bool = field(default=False)
    override: bool = field(default=False)
    load_image: bool = field(default=True)
    clip: bool = field(default=True)


@dataclass
class DatasetFilterConfig:
    scene_pattern: str = '(.*?)'  # All


@dataclass
class UtilityConfig:
    use_validation_for_training: bool = field(default=False)


@dataclass
class GlobalConfig:
    experiment: str
    dataset: DatasetConfig
    eval: TrackerEvalConfig
    object_detection: ObjectDetectionInferenceConfig
    algorithm: TrackerAlgorithmConfig
    dataset_filter: DatasetFilterConfig = field(default_factory=DatasetFilterConfig)
    path: PathConfig = field(default_factory=PathConfig)
    postprocess: TrackerPostprocessConfig = field(default_factory=TrackerPostprocessConfig)
    visualize: TrackerVisualizeConfig = field(default_factory=TrackerVisualizeConfig)
    utility: UtilityConfig = field(default_factory=UtilityConfig)

    @property
    def hash(self) -> str:
        """
        Gets deterministic hash of the crucial tracking setup.

        Returns:
            Deterministic tracking hash.
        """
        return conventions.get_dt_hash(
            dataset_type=self.dataset.type,
            dataset_path=self.dataset.path,
            dataset_params=self.dataset.params,
            algorithm_name=self.algorithm.name,
            algorithm_params=self.algorithm.params,
            object_detection_type=self.object_detection.type,
            object_detection_params=self.object_detection.params,
            object_detection_lookup_path=self.object_detection.lookup_path,
            object_detection_oracle=self.object_detection.oracle,
            scene_pattern=self.dataset_filter.scene_pattern,
            clip=self.eval.clip
        )

    @property
    def run_identifier(self) -> str:
        """
        Gets filesystem-safe run identifier for the current execution.

        Returns:
            Run identifier with `{datetime}_{hash}` format.
        """
        return self._run_identifier

    @property
    def experiment_path(self) -> str:
        """
        Path where tracker results can be found.

        Returns:
            Tracker run directory path.
        """
        return conventions.get_tracker_run_path(
            master_path=self.path.master,
            dataset_type=self.dataset.type,
            dataset_name=self.dataset.name,
            experiment_name=self.experiment,
            split=self.eval.split,
            run_identifier=self.run_identifier
        )

    def __post_init__(self) -> None:
        """
        Postprocess.

        Ignore Returns: This function returns None.
        """
        self.dataset.basepath = os.path.join(self.path.assets, self.dataset.path)
        self.dataset.fullpath = os.path.join(self.dataset.basepath, self.eval.split)
        self._run_datetime = datetime.now().strftime(formats.RUN_DATETIME_FORMAT)
        self._run_identifier = conventions.get_run_identifier(self._run_datetime, self.hash)


# Configuring hydra config store
# If config has `- global_config` in defaults then
# full config is recursively instantiated
cs = ConfigStore.instance()
cs.store(name='global_config', node=GlobalConfig)
