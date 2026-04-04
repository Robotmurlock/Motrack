"""
Pipeline helper functions for tool entrypoints.
"""
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from omegaconf import DictConfig, OmegaConf

from motrack.common import conventions, project, formats
from motrack.config_parser import GlobalConfig
from motrack.utils import rich

logger = logging.getLogger('PipelineUtils')


def task(task_name: str) -> Callable:
    """
    Optional decorator that wraps the task function in extra utilities.

    Args:
        task_name: Task name

    Returns:
        Task wrapper decorator
    """
    def task_wrapper(task_func: Callable) -> Callable:
        """
        Args:
            task_func: Function to wrap

        Returns:
            Wrapped function
        """

        def wrap(cfg: DictConfig):
            # Extracting `master_path` from optional `cfg.path.master`.
            master_path = None
            paths = cfg.get('path')
            if paths is not None:
                master_path = paths.get('master')
            master_path = project.OUTPUTS_PATH if master_path is None else master_path
            Path(master_path).mkdir(parents=True, exist_ok=True)

            # Print config
            rich.print_config_tree(cfg, resolve=True, save_to_file=True)

            # Parse config
            raw_cfg_yaml = OmegaConf.to_yaml(cfg)
            cfg = OmegaConf.to_object(cfg)
            validate_global_config(cfg, task_name=task_name)

            # Store config history
            store_run_history_config(master_path, cfg, raw_cfg_yaml, task_name=task_name)

            # execute the task
            start_time = time.time()

            # Run
            task_func(cfg=cfg)
            logger.info(f"'{task_func.__name__}' execution time: {time.time() - start_time} (s)")

        return wrap

    return task_wrapper


def validate_global_config(cfg: object, task_name: str) -> None:
    """
    Validates that Hydra instantiated the expected config object.

    Args:
        cfg: Parsed Hydra config object.
        task_name: Name of the running task.

    Ignore Returns: This function returns None.

    Raises:
        TypeError: If the parsed config is not an instance of `GlobalConfig`.
    """
    if isinstance(cfg, GlobalConfig):
        return

    received_type = type(cfg).__name__
    raise TypeError(
        f'Task "{task_name}" expects Hydra to instantiate `{GlobalConfig.__name__}`, '
        f'but received `{received_type}` instead. This usually means the tracker config '
        f'is missing `- global_config` in `defaults`, or one of the config groups does '
        f'not match the schema expected by `GlobalConfig`. Please verify that your main '
        f'config includes `- global_config`, and that `dataset`, `eval`, '
        f'`object_detection`, `algorithm`, `postprocess`, `path`, and `visualize` are '
        f'loaded from the expected config groups.'
    )


def store_run_history_config(
    output_dir: str,
    cfg: GlobalConfig,
    raw_cfg_yaml: str,
    task_name: str
) -> None:
    """
    Stores run config of the task run.

    Args:
        output_dir: Task output path
        cfg: Parsed task config
        raw_cfg_yaml: Raw task config serialized as YAML
        task_name: Task name
    """
    tracker_run_path = conventions.get_tracker_run_path(
        master_path=output_dir,
        dataset_type=cfg.dataset.type,
        dataset_name=cfg.dataset.name,
        experiment_name=cfg.experiment,
        split=cfg.eval.split,
        run_identifier=cfg.run_identifier
    )
    config_dirpath = conventions.get_run_configs_path(tracker_run_path)
    dt = datetime.now().strftime(formats.DATETIME_FORMAT)
    config_path = os.path.join(config_dirpath, f'{dt}_{task_name}.yaml')
    Path(config_dirpath).mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(raw_cfg_yaml)
