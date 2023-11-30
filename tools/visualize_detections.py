"""
Visualize detections. Can be used to visually evaluate detector performance without tracking component.
"""
import logging
import os

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from motrack.common.project import CONFIGS_PATH
from motrack.config_parser import GlobalConfig
from motrack.datasets import dataset_factory
from motrack.library.cv.video_writer import MP4Writer
from motrack.object_detection import DetectionManager
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

    scene_names = dataset.scenes
    for scene_name in tqdm(scene_names, desc='Visualize detections', unit='scene'):
        scene_info = dataset.get_scene_info(scene_name)
        scene_length = scene_info.seqlength
        video_path = os.path.join(output_path, f'{scene_name}.mp4')

        with MP4Writer(video_path, fps=cfg.visualize.fps) as video_writer:
            for index in tqdm(range(scene_length), desc=f'Visualizing "{scene_name}"', unit='frame'):
                image = dataset.load_scene_image_by_frame_index(scene_name, index)
                detection_bboxes = detection_manager.predict(scene_name, index)
                for bbox in detection_bboxes:
                    image = bbox.draw(image)

                video_writer.write(image)

@hydra.main(config_path=CONFIGS_PATH, config_name='movesort', version_base='1.1')
def main(cfg: DictConfig):
    # noinspection PyTypeChecker
    inference(cfg)


if __name__ == '__main__':
    main()
