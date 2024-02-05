"""
Creates dataset for FastReid training.

Structure:
{assets}/{dataset-path}/.fast-reid/datasets/{dataset-type}/
    bounding_box_train/
    bounding_box_test/

Each crop has naming format:
`{track_id}_{scene_name}_{frame_id}_crop.jpg`
"""
import logging
import os
from pathlib import Path
import cv2

import hydra
from omegaconf import DictConfig

from motrack.common.project import DANCETRACK_CONFIG_PATH
from motrack.config_parser import GlobalConfig
from motrack.datasets import dataset_factory
from motrack.utils import pipeline
from tqdm import tqdm

logger = logging.getLogger('CreateFastReIDPatches')


FASTREID_PATCHES_DIRNAME = '.fast-reid'
FASTREID_PATCHES_TRAIN_DIRNAME = 'bounding_box_train'
FASTREID_PATCHES_TEST_DIRNAME = 'bounding_box_test'


SEP = '_'
SEP_REPLACEMENT = '+'


@pipeline.task('fastreid-patches')
def run_inference(cfg: GlobalConfig) -> None:
    for split in ['train', 'val']:
        subdir_name = FASTREID_PATCHES_TRAIN_DIRNAME if split == 'train' else FASTREID_PATCHES_TEST_DIRNAME
        subdir_path = os.path.join(cfg.path.assets, cfg.dataset.path, FASTREID_PATCHES_DIRNAME, subdir_name)
        Path(subdir_path).mkdir(parents=True, exist_ok=True)

        logger.info(f'Creating FastReID dataset split "{split}" at path {subdir_path}.')
        dataset = dataset_factory(
            dataset_type=cfg.dataset.type,
            path=os.path.join(cfg.dataset.basepath, split),
            params=cfg.dataset.params
        )

        all_object_ids = [object_id for scene in dataset.scenes for object_id in dataset.get_scene_object_ids(scene)]
        object_id_to_track_id = {object_id: i + 1 for i, object_id in enumerate(all_object_ids)}
        logger.info(f'Total number of unique ids is {len(object_id_to_track_id)}.')

        for scene in tqdm(dataset.scenes, unit='scene', desc='Creating FastReID dataset'):
            scene_name = scene
            if SEP in scene_name:
                scene_name = scene.replace(SEP, SEP_REPLACEMENT)
                logger.warning(f'Scene name "{scene}" contains separator "{SEP}". Renaming it to "{scene_name}".')

            scene_info = dataset.get_scene_info(scene)
            scene_length = scene_info.seqlength
            object_ids = dataset.get_scene_object_ids(scene)

            for frame_index in tqdm(range(scene_length), unit='frame', total=scene_length, desc=f'Creating FastReid for scene {scene}'):
                image = dataset.load_scene_image_by_frame_index(scene, frame_index)

                for object_id in object_ids:
                    data = dataset.get_object_data_by_frame_index(object_id, frame_index)
                    if data is None or data.occ or data.oov:
                        continue

                    bbox = data.create_bbox_object()
                    crop = bbox.crop(image)
                    track_id = object_id_to_track_id[object_id]

                    crop_path = os.path.join(subdir_path, f'{track_id:06d}_{scene_name}_{frame_index + 1:06d}_crop.jpg')
                    cv2.imwrite(crop_path, crop)
                    assert os.path.exists(crop_path), f'Failed to save image "{crop_path}".'


@hydra.main(config_path=DANCETRACK_CONFIG_PATH, config_name='movesort', version_base='1.1')
def main(cfg: DictConfig):
    # noinspection PyTypeChecker
    run_inference(cfg)


if __name__ == '__main__':
    main()
