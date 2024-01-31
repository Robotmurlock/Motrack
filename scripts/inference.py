"""
Tracker inference. Output can directly be evaluated using the TrackEval repo.
"""
import logging
import os
import shutil
from dataclasses import asdict

import hydra
import yaml
from omegaconf import DictConfig

from motrack.common.project import DANCETRACK_CONFIG_PATH
from motrack.config_parser import GlobalConfig
from motrack.datasets import dataset_factory
from motrack.object_detection import DetectionManager
from motrack.tracker import tracker_factory
from motrack.utils import pipeline
from motrack.tools import run_tracker_inference, run_tracker_postprocess

logger = logging.getLogger('TrackerInference')


@pipeline.task('inference')
def run_inference(cfg: GlobalConfig) -> None:
    if os.path.exists(cfg.experiment_path):
        if cfg.eval.override:
            user_input = 'yes'  # `input` from config
        else:
            user_input = input(f'Experiment on path "{cfg.experiment_path}" already exists. '
                               f'Are you sure you want to override it? [yes/no] ').lower()
        if user_input in ['yes', 'y']:
            shutil.rmtree(cfg.experiment_path)
        else:
            logger.info('Aborting!')
            return

    tracker_active_output = os.path.join(cfg.experiment_path, 'active')
    tracker_all_output = os.path.join(cfg.experiment_path, 'all')

    logger.info(f'Saving tracker inference on path "{cfg.experiment_path}".')

    dataset = dataset_factory(
        dataset_type=cfg.dataset.type,
        path=cfg.dataset.fullpath,
        params=cfg.dataset.params,
        test=cfg.eval.split == 'test'
    )

    detection_manager = DetectionManager(
        inference_name=cfg.object_detection.type,
        inference_params=cfg.object_detection.params,
        lookup=cfg.object_detection.load_lookup() if cfg.object_detection.lookup_path is not None else None,
        dataset=dataset,
        cache_path=cfg.object_detection.cache_path,
        oracle=cfg.object_detection.oracle
    )

    tracker = tracker_factory(
        name=cfg.algorithm.name,
        params=cfg.algorithm.params
    )

    run_tracker_inference(
        dataset=dataset,
        tracker=tracker,
        detection_manager=detection_manager,
        tracker_active_output=tracker_active_output,
        tracker_all_output=tracker_all_output,
        clip=cfg.eval.clip,
        scene_pattern=cfg.dataset_filter.scene_pattern,
        load_image=cfg.eval.load_image
    )

    # Save tracker config
    tracker_config_path = os.path.join(cfg.experiment_path, 'config.yaml')
    with open(tracker_config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(asdict(cfg), f)

    if cfg.eval.postprocess:
        logger.info('Performing inference postprocessing...')
        tracker_postprocess_output = os.path.join(cfg.experiment_path, 'postprocess')
        run_tracker_postprocess(
            dataset=dataset,
            tracker_active_output=tracker_active_output,
            tracker_all_output=tracker_all_output,
            tracker_postprocess_output=tracker_postprocess_output,
            postprocess_cfg=cfg.postprocess,
            scene_pattern=cfg.dataset_filter.scene_pattern,
            clip=cfg.eval.clip
        )


@hydra.main(config_path=DANCETRACK_CONFIG_PATH, config_name='movesort', version_base='1.1')
def main(cfg: DictConfig):
    # noinspection PyTypeChecker
    run_inference(cfg)


if __name__ == '__main__':
    main()
