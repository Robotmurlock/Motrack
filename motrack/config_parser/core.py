"""
Tracker config for tool entrypoints.
"""
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore

from motrack.common import conventions, project
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
class TrackerInferenceConfig:
    """
    Controls how tracker inference is executed.

    - split: Dataset split to run on ('train', 'val', or 'test').
    - postprocess: Whether to run offline postprocessing after inference.
    - override: If True, silently overwrite existing experiment outputs.
    - load_image: Whether to load images during inference (disable for speed
      when the tracker/detector doesn't need raw frames).
    - clip: Clip predicted bounding boxes to the image boundaries.
    """
    split: str
    postprocess: bool = field(default=False)
    override: bool = field(default=False)
    load_image: bool = field(default=True)
    clip: bool = field(default=True)


@dataclass
class TrackerEvalConfig:
    """
    Controls how tracker evaluation is executed.

    - eval_output: Which tracker output directory to evaluate. Accepts
      canonical names ('online', 'debug', 'offline') or legacy aliases
      ('active', 'all', 'postprocess').
    - eval_classes: GT class IDs that count as evaluation targets (e.g.
      ``[1]`` for the pedestrian class in MOT Challenge datasets).
    - distractor_classes: GT class IDs whose detections are matched against
      tracker output and removed before scoring. The default covers the
      standard MOT17 distractors (person_on_vehicle=2, static_person=7,
      distractor=8, reflection=12). MOT20 additionally includes
      non_mot_vehicle=6.
    """
    eval_output: str = 'online'
    eval_classes: List[int] = field(default_factory=lambda: [1])
    distractor_classes: List[int] = field(default_factory=lambda: [2, 7, 8, 12])

    def __post_init__(self) -> None:
        self.eval_output = conventions.normalize_tracker_output_name(self.eval_output)


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
    inference: TrackerInferenceConfig
    object_detection: ObjectDetectionInferenceConfig
    algorithm: TrackerAlgorithmConfig
    dataset_filter: DatasetFilterConfig = field(default_factory=DatasetFilterConfig)
    path: PathConfig = field(default_factory=PathConfig)
    postprocess: TrackerPostprocessConfig = field(default_factory=TrackerPostprocessConfig)
    visualize: TrackerVisualizeConfig = field(default_factory=TrackerVisualizeConfig)
    eval: TrackerEvalConfig = field(default_factory=TrackerEvalConfig)
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
            clip=self.inference.clip
        )

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
            split=self.inference.split,
            config_hash=self.hash
        )

    def __post_init__(self) -> None:
        """
        Postprocess.
        """
        self.dataset.basepath = os.path.join(self.path.assets, self.dataset.path)
        self.dataset.fullpath = os.path.join(self.dataset.basepath, self.inference.split)


# Configuring hydra config store
# If config has `- global_config` in defaults then
# full config is recursively instantiated
cs = ConfigStore.instance()
cs.store(name='global_config', node=GlobalConfig)
