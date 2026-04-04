"""
File system conventions used by Motrack.

Tracker results are stored under a deterministic run directory so multiple
configurations can coexist without overwriting each other:

{master_path}/
  {dataset_name}/
    {experiment_name}/
      {split}/
        {run_identifier}/
          online/
            {scene_name}.txt
          debug/
            {scene_name}.txt
          offline/
            {scene_name}.txt
          config.yaml
          run_configs/
            {datetime}_{task_name}.yaml

Definitions:
- master_path: Root directory that contains all saved outputs.
- dataset_name: Human-readable dataset folder name. It falls back to
  `dataset.type` when `dataset.name` is not configured.
- experiment_name: User-selected experiment namespace that keeps runs easy to
  browse.
- split: Dataset split such as `train`, `val`, or `test`.
- run_identifier: Run directory name with format `{datetime}_{hash}`. The
  datetime distinguishes repeated executions of the same configuration, while
  the hash captures the crucial tracking setup.

Tracker outputs use the canonical mode names below:
- online: Results equivalent to the previous `active` output.
- debug: Results equivalent to the previous `all` output.
- offline: Results equivalent to the previous `postprocess` output.

Legacy names are still accepted through `normalize_tracker_output_name(...)` so
existing configs can continue to work while the project migrates to the new
terminology.
"""
import enum
import hashlib
import json
import os
from typing import Optional, Union


ONLINE_DIRNAME = 'online'
DEBUG_DIRNAME = 'debug'
OFFLINE_DIRNAME = 'offline'
RUN_CONFIGS_DIRNAME = 'run_configs'
CONFIG_FILENAME = 'config.yaml'

_OUTPUT_NAME_ALIASES = {
    'active': ONLINE_DIRNAME,
    'all': DEBUG_DIRNAME,
    'postprocess': OFFLINE_DIRNAME,
    ONLINE_DIRNAME: ONLINE_DIRNAME,
    DEBUG_DIRNAME: DEBUG_DIRNAME,
    OFFLINE_DIRNAME: OFFLINE_DIRNAME,
}


class TrackerOutputType(enum.Enum):
    """Canonical tracker output directory names."""

    ONLINE = ONLINE_DIRNAME
    DEBUG = DEBUG_DIRNAME
    OFFLINE = OFFLINE_DIRNAME


def get_dataset_name(dataset_type: str, dataset_name: Optional[str] = None) -> str:
    """
    Gets the filesystem dataset name.

    Args:
        dataset_type: Dataset type used by the factory layer.
        dataset_name: Optional custom dataset folder name.

    Returns:
        Dataset folder name used inside the outputs directory.
    """
    return dataset_type if dataset_name in [None, ''] else dataset_name


def normalize_tracker_output_name(output_name: Union[str, TrackerOutputType]) -> str:
    """
    Normalizes tracker output name to the canonical directory name.

    Args:
        output_name: Canonical or legacy tracker output name.

    Returns:
        Canonical tracker output name.

    Raises:
        ValueError: If the output name is unknown.
    """
    if isinstance(output_name, TrackerOutputType):
        return output_name.value

    normalized_output_name = output_name.lower()
    if normalized_output_name not in _OUTPUT_NAME_ALIASES:
        available = sorted(_OUTPUT_NAME_ALIASES.keys())
        raise ValueError(
            f'Invalid tracker output "{output_name}". Available names: {available}.'
        )

    return _OUTPUT_NAME_ALIASES[normalized_output_name]


def get_dataset_results_path(
    master_path: str,
    dataset_type: str,
    dataset_name: Optional[str] = None
) -> str:
    """
    Gets the dataset output directory path.

    Args:
        master_path: Outputs master path.
        dataset_type: Dataset type used by the pipeline.
        dataset_name: Optional dataset folder override.

    Returns:
        Dataset output directory path.
    """
    return os.path.join(master_path, get_dataset_name(dataset_type, dataset_name))


def get_experiment_results_path(
    master_path: str,
    dataset_type: str,
    experiment_name: str,
    dataset_name: Optional[str] = None
) -> str:
    """
    Gets the experiment output directory path.

    Args:
        master_path: Outputs master path.
        dataset_type: Dataset type used by the pipeline.
        experiment_name: Human-readable experiment name.
        dataset_name: Optional dataset folder override.

    Returns:
        Experiment output directory path.
    """
    return os.path.join(
        get_dataset_results_path(master_path, dataset_type, dataset_name),
        experiment_name
    )


def get_split_results_path(
    master_path: str,
    dataset_type: str,
    experiment_name: str,
    split: str,
    dataset_name: Optional[str] = None
) -> str:
    """
    Gets the split output directory path.

    Args:
        master_path: Outputs master path.
        dataset_type: Dataset type used by the pipeline.
        experiment_name: Human-readable experiment name.
        split: Dataset split.
        dataset_name: Optional dataset folder override.

    Returns:
        Split output directory path.
    """
    return os.path.join(
        get_experiment_results_path(
            master_path=master_path,
            dataset_type=dataset_type,
            experiment_name=experiment_name,
            dataset_name=dataset_name
        ),
        split
    )


def get_dt_hash(
    dataset_type: str,
    dataset_path: str,
    dataset_params: dict,
    algorithm_name: str,
    algorithm_params: dict,
    object_detection_type: str,
    object_detection_params: dict,
    object_detection_lookup_path: Optional[str] = None,
    object_detection_oracle: bool = False,
    scene_pattern: str = '(.*?)',
    clip: bool = True,
    hash_length: int = 12
) -> str:
    """
    Gets deterministic tracking configuration hash.

    Args:
        dataset_type: Dataset type used by the tracker pipeline.
        dataset_path: Dataset path relative to assets root.
        dataset_params: Dataset constructor parameters.
        algorithm_name: Tracker algorithm name.
        algorithm_params: Tracker algorithm parameters.
        object_detection_type: Detector type.
        object_detection_params: Detector constructor parameters.
        object_detection_lookup_path: Optional lookup file path.
        object_detection_oracle: Whether ground-truth detections are used.
        scene_pattern: Regular expression used to filter scenes.
        clip: Whether tracker outputs are clipped to the image size.
        hash_length: Number of hexadecimal characters to return.

    Returns:
        Deterministic short hash for the tracking setup.
    """
    payload = {
        'dataset': {
            'type': dataset_type,
            'path': dataset_path,
            'params': dataset_params,
        },
        'algorithm': {
            'name': algorithm_name,
            'params': algorithm_params,
        },
        'object_detection': {
            'type': object_detection_type,
            'params': object_detection_params,
            'lookup_path': object_detection_lookup_path,
            'oracle': object_detection_oracle,
        },
        'evaluation': {
            'scene_pattern': scene_pattern,
            'clip': clip,
        },
    }
    serialized_payload = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(',', ':')
    )
    digest = hashlib.sha256(serialized_payload.encode('utf-8')).hexdigest()
    return digest[:hash_length]


def get_run_identifier(run_datetime: str, config_hash: str) -> str:
    """
    Gets tracker run identifier.

    Args:
        run_datetime: Run datetime formatted for filesystem usage.
        config_hash: Deterministic tracking configuration hash.

    Returns:
        Run identifier with `{datetime}_{hash}` format.
    """
    return f'{run_datetime}_{config_hash}'


def get_tracker_run_path(
    master_path: str,
    dataset_type: str,
    experiment_name: str,
    split: str,
    run_identifier: str,
    dataset_name: Optional[str] = None
) -> str:
    """
    Gets the tracker run directory path.

    Args:
        master_path: Outputs master path.
        dataset_type: Dataset type used by the pipeline.
        experiment_name: Human-readable experiment name.
        split: Dataset split.
        run_identifier: Run directory name with `{datetime}_{hash}` format.
        dataset_name: Optional dataset folder override.

    Returns:
        Tracker run directory path.
    """
    return os.path.join(
        get_split_results_path(
            master_path=master_path,
            dataset_type=dataset_type,
            experiment_name=experiment_name,
            split=split,
            dataset_name=dataset_name
        ),
        run_identifier
    )


def get_tracker_output_path(
    tracker_run_path: str,
    output_name: Union[str, TrackerOutputType]
) -> str:
    """
    Gets the tracker output directory path.

    Args:
        tracker_run_path: Root directory of the tracker run.
        output_name: Canonical or legacy tracker output name.

    Returns:
        Tracker output directory path.
    """
    return os.path.join(
        tracker_run_path,
        normalize_tracker_output_name(output_name)
    )


def get_run_configs_path(tracker_run_path: str) -> str:
    """
    Gets the stored run-config directory path.

    Args:
        tracker_run_path: Root directory of the tracker run.

    Returns:
        Path where run configuration snapshots are stored.
    """
    return os.path.join(tracker_run_path, RUN_CONFIGS_DIRNAME)


def get_config_snapshot_path(tracker_run_path: str) -> str:
    """
    Gets the tracker config snapshot path.

    Args:
        tracker_run_path: Root directory of the tracker run.

    Returns:
        Tracker config snapshot path.
    """
    return os.path.join(tracker_run_path, CONFIG_FILENAME)


def get_artifact_path(tracker_run_path: str, artifact_name: str) -> str:
    """
    Gets a custom artifact path inside the tracker run directory.

    Args:
        tracker_run_path: Root directory of the tracker run.
        artifact_name: Artifact directory or filename.

    Returns:
        Artifact path inside the tracker run directory.
    """
    return os.path.join(tracker_run_path, artifact_name)
