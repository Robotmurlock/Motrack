"""
Tracker inference.
"""
import logging
import os
import re
from dataclasses import asdict
from typing import List

import hydra
import yaml
from omegaconf import DictConfig
from tqdm import tqdm

from motrack.common.project import CONFIGS_PATH
from motrack.config_parser import GlobalConfig
from motrack.datasets import dataset_factory
from motrack.evaluation.io import TrackerInferenceWriter
from motrack.object_detection import DetectionManager
from motrack.tracker import tracker_factory, Tracklet
from motrack.utils import pipeline
import shutil

logger = logging.getLogger('TrackerEvaluation')


@pipeline.task('inference')
def inference(cfg: GlobalConfig) -> None:
    if os.path.exists(cfg.experiment_path):
        user_input = input(f'Experiment on path "{cfg.experiment_path}" already exists. Are you sure you want to override it? [yes/no] ').lower()
        if user_input in ['yes', 'y']:
            shutil.rmtree(cfg.experiment_path)

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

    scene_names = dataset.scenes
    scene_names = [scene_name for scene_name in scene_names if re.match(cfg.filter.scene_pattern, scene_name)]
    for scene_name in tqdm(scene_names, desc='Simulating tracker', unit='scene'):
        scene_info = dataset.get_scene_info(scene_name)
        scene_length = scene_info.seqlength
        imheight = scene_info.imheight
        imwidth = scene_info.imwidth

        tracker = tracker_factory(
            name=cfg.algorithm.name,
            params=cfg.algorithm.params
        )

        clip = cfg.dataset.type != 'MOT17'
        with TrackerInferenceWriter(tracker_active_output, scene_name, image_height=imheight, image_width=imwidth, clip=clip) as tracker_active_inf_writer, \
            TrackerInferenceWriter(tracker_all_output, scene_name, image_height=imheight, image_width=imwidth, clip=clip) as tracker_all_inf_writer:
            tracklets: List[Tracklet] = []
            for index in tqdm(range(scene_length), desc=f'Simulating "{scene_name}"', unit='frame'):
                # Perform OD inference
                detection_bboxes = detection_manager.predict(scene_name, index)

                # Perform tracking step
                active_tracklets, tracklets = tracker.track(
                    tracklets=tracklets,
                    detections=detection_bboxes,
                    frame_index=index + 1  # Counts from 1 instead of 0
                )

                # Save inference
                for tracklet in active_tracklets:
                    tracker_active_inf_writer.write(index, tracklet)

                for tracklet in tracklets:
                    tracker_all_inf_writer.write(index, tracklet)


    # Save tracker config
    tracker_config_path = os.path.join(cfg.experiment_path, 'config.yaml')
    with open(tracker_config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(asdict(cfg), f)

@hydra.main(config_path=CONFIGS_PATH, config_name='movesort', version_base='1.1')
def main(cfg: DictConfig):
    # noinspection PyTypeChecker
    inference(cfg)


if __name__ == '__main__':
    main()
