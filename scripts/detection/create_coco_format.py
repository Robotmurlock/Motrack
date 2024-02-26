"""
Creates dataset COCO annotations for YOLOX training.

Structure:
{assets}/{dataset-path}/.coco/
    {split}.json
"""
import json
import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from motrack.common.project import DANCETRACK_CONFIG_PATH
from motrack.config_parser import GlobalConfig
from motrack.datasets import dataset_factory
from motrack.utils import pipeline

logger = logging.getLogger('CreateObjectDetectionCOCOFormat')


COCO_DIRNAME = '.coco'
ANNOTATIONS_DIRNAME = 'annotations'
CATEGORY_ID = 1
CATEGORY_NAME = 'pedestrian'


@pipeline.task('coco-annotations')
def run_inference(cfg: GlobalConfig) -> None:
    coco_path = os.path.join(cfg.dataset.basepath, COCO_DIRNAME)
    annotations_path = os.path.join(coco_path, ANNOTATIONS_DIRNAME)
    Path(annotations_path).mkdir(parents=True, exist_ok=True)

    if cfg.utility.use_validation_for_training:
        logger.warning('Including validation dataset in training is not supported!')

    for split in ['train', 'val']:
        split_annotations_path = os.path.join(annotations_path, f'{split}.json')
        annotations = {
            'images': [],
            'annotations': [],
            'videos': [],
            'categories': [
                {'id': CATEGORY_ID, 'name': CATEGORY_NAME}  # FIXME: Currently only supports dataset with a single category
            ]
        }

        logger.info(f'Creating COCO dataset split "{split}" at the path "{split_annotations_path}".')
        dataset_split_path = os.path.join(cfg.dataset.basepath, split)
        dataset = dataset_factory(
            dataset_type=cfg.dataset.type,
            path=dataset_split_path,
            params=cfg.dataset.params
        )

        frame_global_index = 1
        ann_global_index = 1

        for scene_index, scene in tqdm(enumerate(dataset.scenes), unit='scene', desc='Creating COCO annotations'):
            scene_info = dataset.get_scene_info(scene)
            scene_length = scene_info.seqlength
            object_ids = dataset.get_scene_object_ids(scene)
            annotations['videos'].append(scene_index + 1)

            for frame_index in tqdm(range(scene_length), unit='frame', total=scene_length, desc=f'Creating COCO annotations for scene {scene}'):
                image_filepath = dataset.get_scene_image_path(scene, frame_index)
                image_info = {
                    'file_name': os.path.relpath(image_filepath, coco_path),
                    'id': frame_global_index,
                    'frame_id': frame_index + 1,
                    'prev_image_id': frame_index if frame_index > 0 else -1,
                    'next_image_id': frame_index + 2 if frame_index + 1 == scene_info.seqlength else -1,
                    'video_id': scene_index + 1,
                    'height': scene_info.imheight,
                    'width': scene_info.imwidth
                }
                frame_global_index += 1
                annotations['images'].append(image_info)

                for object_id in object_ids:
                    data = dataset.get_object_data_by_frame_index(object_id, frame_index)
                    if data is None or data.occ or data.oov:
                        continue

                    bbox = data.create_bbox_object()
                    ann = {
                        'id': ann_global_index,
                        'category_id': CATEGORY_ID,
                        'image_id': frame_index + 1,
                        'track_id': data.track_id,
                        'bbox': bbox.as_numpy_xywh().tolist(),
                        'conf': 1.0,  # weak labels are not supported
                        'iscrowd': 0,
                        'area': bbox.area
                    }
                    annotations['annotations'].append(ann)
                    ann_global_index += 1

        logger.info(f'Split "{split}" has total number of {len(annotations["images"])} images and {len(annotations["annotations"])} annotations.')

        with open(split_annotations_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2)


@hydra.main(config_path=DANCETRACK_CONFIG_PATH, config_name='movesort', version_base='1.1')
def main(cfg: DictConfig):
    # noinspection PyTypeChecker
    run_inference(cfg)


if __name__ == '__main__':
    main()
