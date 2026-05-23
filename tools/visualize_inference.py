"""
Tracker inference visualization.
"""
import logging
import os

import hydra
from motrack.common import conventions
from motrack.common.project import DANCETRACK_TRACKERS_CONFIG_PATH
from motrack.config_parser import GlobalConfig
from motrack.datasets import dataset_factory
from motrack.tools import run_visualize_tracker_inference
from motrack.utils import pipeline
from omegaconf import DictConfig

logger = logging.getLogger('Tool-TrackerVisualization')


@pipeline.task('visualize-inference')
def visualize_inference(cfg: GlobalConfig) -> None:
    tracker_output_option = conventions.get_tracker_output_path(
        cfg.experiment_path,
        cfg.visualize.option
    )
    tracker_output_online = conventions.get_tracker_output_path(
        cfg.experiment_path,
        conventions.TrackerOutputType.ONLINE
    )
    assert os.path.exists(cfg.experiment_path), f'Path "{cfg.experiment_path}" does not exist!'
    logger.info(f'Visualizing tracker inference on path "{tracker_output_option}".')

    dataset = dataset_factory(
        dataset_type=cfg.dataset.type,
        path=cfg.dataset.fullpath,
        params=cfg.dataset.params,
        test=cfg.inference.split == 'test'
    )

    run_visualize_tracker_inference(
        dataset=dataset,
        tracker_active_output=tracker_output_online,
        tracker_output_option=tracker_output_option,
        visualize_cfg=cfg.visualize,
        scene_pattern=cfg.dataset_filter.scene_pattern
    )


@hydra.main(config_path=DANCETRACK_TRACKERS_CONFIG_PATH, config_name='movesort', version_base='1.1')
def main(cfg: DictConfig):
    # noinspection PyTypeChecker
    visualize_inference(cfg)


if __name__ == '__main__':
    main()
