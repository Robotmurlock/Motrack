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
    Makes multirun more resistant to failure.
    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir

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
            # Extracting `output_dir` from optional `cfg.paths.output_dir`.
            output_dir = None
            paths = cfg.get('paths')
            if paths is not None:
                output_dir = paths.get('master')
            output_dir = project.OUTPUTS_PATH if output_dir is None else output_dir
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Print config
            rich.print_config_tree(cfg, resolve=True, save_to_file=True)

            # Store config history
            store_run_history_config(output_dir, cfg, task_name=task_name)

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
    config_dirpath = os.path.join(output_dir, cfg.dataset_name, cfg.experiment_name)
    dt = datetime.now().strftime(formats.DATETIME_FORMAT)
    config_path = os.path.join(config_dirpath, f'{dt}_{task_name}.yaml')
    Path(config_dirpath).mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(OmegaConf.to_yaml(cfg))
