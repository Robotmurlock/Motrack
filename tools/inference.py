"""
Tracker inference. Output can directly be evaluated using the TrackEval repo.
"""
from dataclasses import asdict
from datetime import datetime
import logging
import os
import shutil
from typing import Optional

import yaml

import hydra
from motrack.common import conventions
from motrack.common.project import DANCETRACK_TRACKERS_CONFIG_PATH
from motrack.config_parser import GlobalConfig
from motrack.datasets import dataset_factory
from motrack.object_detection import DetectionManager
from motrack.tools import run_tracker_inference, run_tracker_postprocess
from motrack.tracker import tracker_factory
from motrack.utils import pipeline
from tools.data import InferenceOutputData

logger = logging.getLogger('Tool-TrackerInference')


def _run_inference_inner(cfg: GlobalConfig, inference_output: Optional[InferenceOutputData] = None) -> None:
    """Core inference logic. Called by both the CLI wrapper and the optimizer."""
    if os.path.exists(cfg.experiment_path):
        if cfg.inference.override:
            user_input = 'yes'  # `input` from config
        else:
            user_input = input(f'Experiment on path "{cfg.experiment_path}" already exists. '
                               f'Are you sure you want to override it? [yes/no] ').lower()
        if user_input in ['yes', 'y']:
            shutil.rmtree(cfg.experiment_path)
        else:
            logger.info('Aborting!')
            return

    tracker_online_output = conventions.get_tracker_output_path(
        cfg.experiment_path,
        conventions.TrackerOutputType.ONLINE
    )
    tracker_debug_output = conventions.get_tracker_output_path(
        cfg.experiment_path,
        conventions.TrackerOutputType.DEBUG
    )

    logger.info(f'Saving tracker inference on path "{cfg.experiment_path}".')

    dataset = dataset_factory(
        dataset_type=cfg.dataset.type,
        path=cfg.dataset.fullpath,
        params=cfg.dataset.params,
        test=cfg.inference.split == 'test'
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
        tracker_active_output=tracker_online_output,
        tracker_all_output=tracker_debug_output,
        clip=cfg.inference.clip,
        scene_pattern=cfg.dataset_filter.scene_pattern,
        load_image=cfg.inference.load_image
    )

    tracker_config_path = conventions.get_config_snapshot_path(cfg.experiment_path)
    with open(tracker_config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(asdict(cfg), f)

    inference_output_path = conventions.get_run_meta_path(cfg.experiment_path)
    if inference_output is None:
        inference_output = InferenceOutputData(created_at=datetime.now().isoformat())
    inference_output.save(inference_output_path)

    if cfg.inference.postprocess:
        logger.info('Performing inference postprocessing...')
        tracker_offline_output = conventions.get_tracker_output_path(
            cfg.experiment_path,
            conventions.TrackerOutputType.OFFLINE
        )
        run_tracker_postprocess(
            dataset=dataset,
            tracker_active_output=tracker_online_output,
            tracker_all_output=tracker_debug_output,
            tracker_postprocess_output=tracker_offline_output,
            postprocess_cfg=cfg.postprocess,
            scene_pattern=cfg.dataset_filter.scene_pattern,
            clip=cfg.inference.clip
        )


@hydra.main(config_path=DANCETRACK_TRACKERS_CONFIG_PATH, config_name='movesort', version_base='1.1')
@pipeline.task('inference')
def main(cfg: GlobalConfig) -> None:
    _run_inference_inner(cfg)


if __name__ == '__main__':
    main()
