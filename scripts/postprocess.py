"""
Tracker inference postprocess (offline). Output can directly be evaluated using the TrackEval repo.
"""
import logging
import os

import hydra
from omegaconf import DictConfig

from motrack.common.project import CONFIGS_PATH
from motrack.config_parser import GlobalConfig
from motrack.datasets import dataset_factory
from motrack.tools.postprocess import run_tracker_postprocess
from motrack.utils import pipeline

logger = logging.getLogger('TrackerVizualization')


@pipeline.task('postprocess')
def run_postprocess(cfg: GlobalConfig) -> None:
    assert os.path.exists(cfg.experiment_path), f'Path "{cfg.experiment_path}" does not exist!'
    logger.info(f'Postprocessing tracker inference on path "{cfg.experiment_path}".')

    tracker_all_output = os.path.join(cfg.experiment_path, 'all')
    tracker_active_output = os.path.join(cfg.experiment_path, 'active')
    tracker_postprocess_output = os.path.join(cfg.experiment_path, 'postprocess')

    dataset = dataset_factory(
        dataset_type=cfg.dataset.type,
        path=cfg.dataset.fullpath,
        params=cfg.dataset.params,
        test=cfg.eval.split == 'test'
    )

    run_tracker_postprocess(
        dataset=dataset,
        tracker_active_output=tracker_active_output,
        tracker_all_output=tracker_all_output,
        tracker_postprocess_output=tracker_postprocess_output,
        postprocess_cfg=cfg.postprocess,
        scene_pattern=cfg.dataset_filter.scene_pattern
    )


@hydra.main(config_path=CONFIGS_PATH, config_name='movesort', version_base='1.1')
def main(cfg: DictConfig):
    # noinspection PyTypeChecker
    run_postprocess(cfg)


if __name__ == '__main__':
    main()
