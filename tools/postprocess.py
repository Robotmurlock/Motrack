"""
Tracker inference postprocess.

TODO: Refactor
"""
import logging
import os
import re
from collections import Counter, defaultdict
from typing import List, Dict

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from motrack.common.project import CONFIGS_PATH
from motrack.config_parser import GlobalConfig
from motrack.datasets.factory import dataset_factory
from motrack.evaluation.io import TrackerInferenceWriter, TrackerInferenceReader
from motrack.library.cv.bbox import PredBBox, Point
from motrack.tracker import Tracklet
from motrack.utils import pipeline

logger = logging.getLogger('TrackerVizualization')


INF = 999_999


def element_distance_from_list(query: int, keys: List[int]) -> int:
    min_distance = None
    for key in keys:
        distance = abs(query - key)
        min_distance = distance if min_distance is None else min(distance, min_distance)

    return min_distance


def find_closest_prev_element(query: int, keys: List[int]) -> int:
    value = None
    for key in keys:
        if key > query:
            break
        value = key

    assert value is not None
    return value


def find_closest_next_element(query: int, keys: List[int]) -> int:
    value = None
    for key in keys:
        if key < query:
            continue
        value = key
        break

    assert value is not None
    return value


def interpolate_bbox(start_index: int, start_bbox: PredBBox, end_index: int, end_bbox: PredBBox, index: int) -> PredBBox:
    """
    Perform linear interpolation between two bounding boxes at a specific index.

    Args:
        start_index: The frame index of the starting bounding box.
        start_bbox: The starting bounding box.
        end_index: The frame index of the ending bounding box.
        end_bbox: The ending bounding box.
        index: The target frame index for interpolation.

    Returns:
        Interpolated bounding box at the target index.
    """
    assert start_index < index < end_index, "The target index must be between start_index and end_index"

    # Calculate alpha based on the target index
    alpha = (index - start_index) / (end_index - start_index)

    # Interpolate upper left corner
    interpolated_upper_left = Point(
        start_bbox.upper_left.x + alpha * (end_bbox.upper_left.x - start_bbox.upper_left.x),
        start_bbox.upper_left.y + alpha * (end_bbox.upper_left.y - start_bbox.upper_left.y)
    )

    # Interpolate bottom right corner
    interpolated_bottom_right = Point(
        start_bbox.bottom_right.x + alpha * (end_bbox.bottom_right.x - start_bbox.bottom_right.x),
        start_bbox.bottom_right.y + alpha * (end_bbox.bottom_right.y - start_bbox.bottom_right.y)
    )

    # Interpolate label and confidence (if available)
    interpolated_label = start_bbox.label
    interpolated_conf = None
    if start_bbox.conf is not None and end_bbox.conf is not None:
        interpolated_conf = start_bbox.conf + alpha * (end_bbox.conf - start_bbox.conf)

    interpolated_bbox = PredBBox(
        upper_left=interpolated_upper_left,
        bottom_right=interpolated_bottom_right,
        label=interpolated_label,
        conf=interpolated_conf
    )

    return interpolated_bbox


@pipeline.task('postprocess')
def postprocess(cfg: GlobalConfig) -> None:
    assert os.path.exists(cfg.experiment_path), f'Path "{cfg.experiment_path}" does not exist!'
    logger.info(f'Postprocessing tracker inference on path "{cfg.experiment_path}".')

    tracker_all_output = os.path.join(cfg.experiment_path, 'all')
    tracker_active_output = os.path.join(cfg.experiment_path, 'active')
    tracker_postprocess_output = os.path.join(cfg.experiment_path, 'postprocess')

    dataset = dataset_factory(
        name=cfg.dataset.name,
        path=cfg.dataset.fullpath,
        params=cfg.dataset.params
    )

    scene_names = dataset.scenes
    scene_names = [scene_name for scene_name in scene_names if re.match(cfg.filter.scene_pattern, scene_name)]
    for scene_name in tqdm(scene_names, desc='Postprocessing tracker', unit='scene'):
        scene_info = dataset.get_scene_info(scene_name)
        scene_length = scene_info.seqlength
        imheight = scene_info.imheight
        imwidth = scene_info.imwidth

        tracklet_presence_counter = Counter()  # Used to visualize new, appearing tracklets
        tracklet_frame_bboxes: Dict[str, Dict[int, PredBBox]] = defaultdict(dict)
        with TrackerInferenceReader(tracker_active_output, scene_name, image_height=imheight, image_width=imwidth) as tracker_inf_reader:
            last_read = tracker_inf_reader.read()

            for index in tqdm(range(scene_length), desc=f'Counting occurrences "{scene_name}"', unit='frame'):
                if last_read is not None and index == last_read.frame_index:
                    for tracklet_id, bbox in last_read.objects.items():
                        tracklet_presence_counter[tracklet_id] += 1
                        tracklet_frame_bboxes[tracklet_id][index] = bbox

                    last_read = tracker_inf_reader.read()

        # (1) Filter short tracklets (they have less than X active frames)
        tracklets_to_keep = {k for k, v in tracklet_presence_counter.items() if v >= cfg.postprocess.min_tracklet_length}
        tracklet_frame_bboxes = dict(tracklet_frame_bboxes)

        clip = cfg.dataset.name != 'MOT17'
        with TrackerInferenceReader(tracker_all_output, scene_name, image_height=imheight, image_width=imwidth) as tracker_all_inf_reader, \
            TrackerInferenceWriter(tracker_postprocess_output, scene_name, image_height=imheight, image_width=imwidth, clip=clip) as tracker_inf_writer:

            last_all_read = tracker_all_inf_reader.read()

            for index in tqdm(range(scene_length), desc=f'Postprocessing "{scene_name}"', unit='frame'):
                if last_all_read is not None and index == last_all_read.frame_index:
                    for tracklet_id, bbox in last_all_read.objects.items():

                        if tracklet_id not in tracklets_to_keep:
                            # Marked as FP in postprocessing
                            continue

                        tracklet_indices = list(tracklet_frame_bboxes[tracklet_id].keys())
                        keep = False
                        if index in tracklet_indices:
                            keep = True

                        # (2) Linear interpolation
                        if index not in tracklet_indices and min(tracklet_indices) <= index <= max(tracklet_indices):
                            prev_index = find_closest_prev_element(index, tracklet_indices)
                            next_index = find_closest_next_element(index, tracklet_indices)
                            if next_index - prev_index > cfg.postprocess.linear_interpolation_threshold:
                                continue

                            bbox = interpolate_bbox(
                                start_index=prev_index,
                                start_bbox=tracklet_frame_bboxes[tracklet_id][prev_index],
                                end_index=next_index,
                                end_bbox=tracklet_frame_bboxes[tracklet_id][next_index],
                                index=index
                            )
                            keep = True

                        # (3) Add trajectory initialization
                        start_index = min(tracklet_indices)
                        if index < start_index and start_index - index <= cfg.postprocess.init_threshold:
                            keep = True

                        if keep:
                            tracklet = Tracklet(bbox=bbox, frame_index=index, _id=int(tracklet_id))
                            tracker_inf_writer.write(frame_index=index, tracklet=tracklet)

                    last_all_read = tracker_all_inf_reader.read()


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    # noinspection PyTypeChecker
    postprocess(cfg)


if __name__ == '__main__':
    main()
