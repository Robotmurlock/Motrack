"""
Creates dataset YOLO annotations for YOLOv8 training.
Tutorial: https://www.kaggle.com/code/momiradzemovic/animal-detection-yolov8

Structure:
yolo-dataset/
    train/
        images/
            {id_1}.jpg
            {id_2}.jpg
            ...
        labels/
            {id_1}.txt
            {id_2}.txt
            ...
        ...
    test/
    ...
"""
import logging
import os
from pathlib import Path
from typing import List
import yaml

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from motrack.common.project import DANCETRACK_CONFIG_PATH
from motrack.config_parser import GlobalConfig
from motrack.datasets import dataset_factory
from motrack.utils import pipeline

logger = logging.getLogger('CreateObjectDetectionCOCOFormat')


YOLO_DIRNAME = '.yolo'
LEAK_SUFFIX = '-leak'
IMAGES_DIRNAME = 'images'
LABELS_DIRNAME = 'labels'
CATEGORY_INDEX = 0
CATEGORY_NAME = 'pedestrian'


def save_labels(labels: List[List[float]], filepath: str) -> None:
    """
    Saves labels as a csv-like txt file.

    Args:
        labels: List of labels
        filepath: Path where the labels are stored

    Returns:

    """
    with open(filepath, 'w', encoding='utf-8') as f:
        label_lines = [' '.join([str(v) for v in label]) for label in labels]
        f.write('\n'.join(label_lines))
        f.write('\n')


@pipeline.task('yolo-annotations')
def run_inference(cfg: GlobalConfig) -> None:
    dataset_dirname = YOLO_DIRNAME if not cfg.utility.use_validation_for_training else f'{YOLO_DIRNAME}{LEAK_SUFFIX}'
    yolo_dataset_path = os.path.join(cfg.dataset.basepath, dataset_dirname)
    Path(yolo_dataset_path).mkdir(parents=True, exist_ok=True)
    frame_global_index = 1

    # Used in case `cfg.utility.use_validation_for_training == True`
    train_images_path = os.path.join(yolo_dataset_path, 'train', IMAGES_DIRNAME)
    train_labels_path = os.path.join(yolo_dataset_path, 'train', LABELS_DIRNAME)

    for split in ['train', 'val']:
        yolo_dataset_split_path = os.path.join(yolo_dataset_path, split)
        logger.info(f'Creating YOLO dataset split "{split}" at the path "{yolo_dataset_split_path}".')
        images_path = os.path.join(yolo_dataset_split_path, IMAGES_DIRNAME)
        labels_path = os.path.join(yolo_dataset_split_path, LABELS_DIRNAME)
        Path(images_path).mkdir(parents=True, exist_ok=True)
        Path(labels_path).mkdir(parents=True, exist_ok=True)

        dataset_split_path = os.path.join(cfg.dataset.basepath, split)
        dataset = dataset_factory(
            dataset_type=cfg.dataset.type,
            path=dataset_split_path,
            params=cfg.dataset.params
        )

        for scene_index, scene in tqdm(enumerate(dataset.scenes), unit='scene', desc='Creating YOLO annotations'):
            scene_info = dataset.get_scene_info(scene)
            scene_length = scene_info.seqlength
            object_ids = dataset.get_scene_object_ids(scene)

            for frame_index in tqdm(range(scene_length), unit='frame', total=scene_length, desc=f'Creating YOLO annotations for scene {scene}'):
                image_filepath = dataset.get_scene_image_path(scene, frame_index)
                image_id = f'{frame_global_index:08d}'
                yolo_dataset_image_filepath = os.path.join(images_path, f'{image_id}.jpg')
                yolo_dataset_labels_filepath = os.path.join(labels_path, f'{image_id}.txt')

                labels: List[list] = []
                for object_id in object_ids:
                    data = dataset.get_object_data_by_frame_index(object_id, frame_index)
                    if data is None or data.occ or data.oov:
                        continue

                    bbox = data.create_bbox_object()
                    labels.append([CATEGORY_INDEX, bbox.center.x, bbox.center.y, bbox.width, bbox.height])

                save_labels(labels, yolo_dataset_labels_filepath)
                os.symlink(image_filepath, yolo_dataset_image_filepath)

                if split == 'val' and cfg.utility.use_validation_for_training:
                    # Save validation dataset in training
                    # Warning: This detector can't be used for tracker hyperparameter tuning on validation dataset!
                    yolo_dataset_train_images_filepath = os.path.join(train_images_path, f'{image_id}.jpg')
                    yolo_dataset_train_labels_filepath = os.path.join(train_labels_path, f'{image_id}.txt')

                    save_labels(labels, yolo_dataset_train_labels_filepath)
                    os.symlink(image_filepath, yolo_dataset_train_images_filepath)

                frame_global_index += 1

    config = {
        'path': yolo_dataset_path,
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,
        'names': [CATEGORY_NAME]
    }
    yolo_dataset_config_path = os.path.join(yolo_dataset_path, 'config.yaml')
    with open(yolo_dataset_config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f)

    logger.info(f'Saved YOLO dataset config at path "{yolo_dataset_config_path}".')


@hydra.main(config_path=DANCETRACK_CONFIG_PATH, config_name='movesort', version_base='1.1')
def main(cfg: DictConfig):
    # noinspection PyTypeChecker
    run_inference(cfg)


if __name__ == '__main__':
    main()
