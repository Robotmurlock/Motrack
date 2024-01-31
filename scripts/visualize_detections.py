"""
Visualize detections. Can be used to visually evaluate detector performance without tracking component.
"""
import logging
import os

import hydra
from omegaconf import DictConfig

from motrack.common.project import DANCETRACK_CONFIG_PATH
from motrack.config_parser import GlobalConfig
from motrack.datasets import dataset_factory
from motrack.object_detection import DetectionManager
from motrack.tools import run_visualize_detections
from motrack.utils import pipeline

logger = logging.getLogger('VisualizeDetections')


@pipeline.task('visualize-detections')
def inference(cfg: GlobalConfig) -> None:
    output_path = os.path.join(cfg.experiment_path, 'visualize_detections')
    logger.info(f'Saving detection visualizations at "{output_path}".')

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

    run_visualize_detections(
        dataset=dataset,
        detection_manager=detection_manager,
        output_path=output_path,
        fps=cfg.visualize.fps
    )



@hydra.main(config_path=DANCETRACK_CONFIG_PATH, config_name='movesort', version_base='1.1')
def main(cfg: DictConfig):
    # noinspection PyTypeChecker
    inference(cfg)


if __name__ == '__main__':
    main()
