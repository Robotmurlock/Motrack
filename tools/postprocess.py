"""
Tracker inference postprocess (offline). Output can directly be evaluated using the TrackEval repo.
"""
import logging
import os

import hydra
from motrack.common import conventions
from motrack.common.project import DANCETRACK_TRACKERS_CONFIG_PATH
from motrack.config_parser import GlobalConfig
from motrack.datasets import dataset_factory
from motrack.tools.postprocess import run_tracker_postprocess
from motrack.utils import pipeline
from omegaconf import DictConfig

logger = logging.getLogger('Tool-TrackerVisualization')


@pipeline.task('postprocess')
def run_postprocess(cfg: GlobalConfig) -> None:
    assert os.path.exists(cfg.experiment_path), f'Path "{cfg.experiment_path}" does not exist!'
    logger.info(f'Postprocessing tracker inference on path "{cfg.experiment_path}".')

    tracker_debug_output = conventions.get_tracker_output_path(
        cfg.experiment_path,
        conventions.TrackerOutputType.DEBUG
    )
    tracker_online_output = conventions.get_tracker_output_path(
        cfg.experiment_path,
        conventions.TrackerOutputType.ONLINE
    )
    tracker_offline_output = conventions.get_tracker_output_path(
        cfg.experiment_path,
        conventions.TrackerOutputType.OFFLINE
    )

    dataset = dataset_factory(
        dataset_type=cfg.dataset.type,
        path=cfg.dataset.fullpath,
        params=cfg.dataset.params,
        test=cfg.eval.split == 'test'
    )

    run_tracker_postprocess(
        dataset=dataset,
        tracker_active_output=tracker_online_output,
        tracker_all_output=tracker_debug_output,
        tracker_postprocess_output=tracker_offline_output,
        postprocess_cfg=cfg.postprocess,
        scene_pattern=cfg.dataset_filter.scene_pattern,
        clip=cfg.eval.clip
    )


@hydra.main(config_path=DANCETRACK_TRACKERS_CONFIG_PATH, config_name='movesort', version_base='1.1')
def main(cfg: DictConfig):
    # noinspection PyTypeChecker
    run_postprocess(cfg)


if __name__ == '__main__':
    main()
