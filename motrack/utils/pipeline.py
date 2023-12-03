"""
Pipeline (tools scripts) functions.
"""
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from omegaconf import DictConfig, OmegaConf

from motrack.common import project, formats
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

            # Store config history
            store_run_history_config(master_path, cfg, task_name=task_name)

            # execute the task
            start_time = time.time()
            # Parse config
            cfg = OmegaConf.to_object(cfg)

            # Run
            task_func(cfg=cfg)
            logger.info(f"'{task_func.__name__}' execution time: {time.time() - start_time} (s)")

        return wrap

    return task_wrapper


def store_run_history_config(output_dir: str, cfg: DictConfig, task_name: str) -> None:
    """
    Stores run config of the task run.

    Args:
        output_dir: Task output path
        cfg: Task config
        task_name: Task name
    """
    config_dirpath = os.path.join(output_dir, cfg.dataset.type, cfg.algorithm.name, cfg.experiment, 'run_configs')
    dt = datetime.now().strftime(formats.DATETIME_FORMAT)
    config_path = os.path.join(config_dirpath, f'{dt}_{task_name}.yaml')
    Path(config_dirpath).mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(OmegaConf.to_yaml(cfg))
