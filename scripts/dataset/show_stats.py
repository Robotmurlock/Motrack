"""
Tracker inference. Output can directly be evaluated using the TrackEval repo.
"""
import logging
import os

import hydra
from omegaconf import DictConfig

from motrack.common.project import DANCETRACK_CONFIG_PATH
from motrack.config_parser import GlobalConfig
from motrack.datasets import dataset_factory, BaseDataset
from motrack.utils import pipeline

logger = logging.getLogger('Script-DatasetShowStats')


def calculate_split_stats(dataset: BaseDataset) -> dict:
    stats = {
        'number_of_scenes': 0,
        'number_of_frames': 0,
        'number_of_boxes': 0,
        'number_of_tracks': 0,
        'number_of_annotated_frames': 0
    }

    scenes = dataset.scenes
    for scene in scenes:
        scene_info = dataset.get_scene_info(scene)
        scene_object_ids = dataset.get_scene_object_ids(scene)

        stats['number_of_scenes'] += 1
        stats['number_of_frames'] += scene_info.seqlength
        stats['number_of_tracks'] += len(scene_object_ids)


        for frame_index in range(scene_info.seqlength):
            annotated = False
            for object_id in scene_object_ids:
                data = dataset.get_object_data_by_frame_index(object_id, frame_index)
                if data is None:
                    continue

                stats['number_of_boxes'] += 1
                annotated = True

            if annotated:
                stats['number_of_annotated_frames'] += 1

    stats['average_number_of_objects_in_frames'] = stats['number_of_boxes'] / stats['number_of_annotated_frames']
    return stats


@pipeline.task('dataset-show-stats')
def run_dataset_show_stats(cfg: GlobalConfig) -> None:
    for split in ['train', 'val', 'test']:
        dataset_split_path = os.path.join(cfg.dataset.basepath, split)
        dataset = dataset_factory(
            dataset_type=cfg.dataset.type,
            path=dataset_split_path,
            params=cfg.dataset.params,
            test=False
        )

        split_stats = calculate_split_stats(dataset)
        print(split_stats)


@hydra.main(config_path=DANCETRACK_CONFIG_PATH, config_name='movesort', version_base='1.1')
def main(cfg: DictConfig):
    # noinspection PyTypeChecker
    run_dataset_show_stats(cfg)


if __name__ == '__main__':
    main()
