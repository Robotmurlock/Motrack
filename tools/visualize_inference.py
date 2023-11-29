"""
Tracker inference visualization.
"""
import logging
import os
import re
from collections import Counter

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from motrack.common.project import CONFIGS_PATH
from motrack.config_parser import GlobalConfig
from motrack.datasets.factory import dataset_factory
from motrack.evaluation.io import TrackerInferenceReader
from motrack.library.cv import color_palette
from motrack.library.cv.bbox import PredBBox, BBox
from motrack.library.cv.drawing import draw_text
from motrack.library.cv.video_writer import MP4Writer
from motrack.utils import pipeline

logger = logging.getLogger('TrackerVizualization')


def draw_tracklet(
    frame: np.ndarray,
    tracklet_id: str,
    tracklet_age: int,
    bbox: PredBBox,
    new: bool = False,
    active: bool = True
) -> np.ndarray:
    """
    Draw tracklet on the frame.

    Args:
        frame: Frame
        tracklet_id: Tracklet id
        tracklet_age: How long does the tracklet exist
        bbox: BBox
        new: Is the tracking object new (new id)
            - New object has thick bbox
        active: Is tracklet active or not

    Returns:
        Frame with drawn tracklet info
    """
    color = color_palette.ALL_COLORS_EXPECT_BLACK[int(tracklet_id) % len(color_palette.ALL_COLORS_EXPECT_BLACK)] \
        if active else color_palette.BLACK
    thickness = 5 if new else 2
    frame = BBox.draw(bbox, frame, color=color, thickness=thickness)
    left, top, _, _ = bbox.scaled_xyxy_from_image(frame)
    text = f'id={tracklet_id}, age={tracklet_age}, conf={100 * bbox.conf:.0f}%'
    return draw_text(frame, text, round(left), round(top), color=color)


@pipeline.task('visualize-inference')
def visualize_inference(cfg: GlobalConfig) -> None:
    tracker_output_option = os.path.join(cfg.experiment_path, cfg.visualize.option)
    tracker_output_active = os.path.join(cfg.experiment_path, 'active')
    assert os.path.exists(cfg.experiment_path), f'Path "{cfg.experiment_path}" does not exist!'
    logger.info(f'Visualizing tracker inference on path "{tracker_output_option}".')

    dataset = dataset_factory(
        dataset_type=cfg.dataset.type,
        path=cfg.dataset.fullpath,
        params=cfg.dataset.params,
        test=cfg.eval.split == 'test'
    )

    scene_names = dataset.scenes
    scene_names = [scene_name for scene_name in scene_names if re.match(cfg.filter.scene_pattern, scene_name)]
    for scene_name in tqdm(scene_names, desc='Visualizing tracker', unit='scene'):
        scene_info = dataset.get_scene_info(scene_name)
        scene_length = scene_info.seqlength
        imheight = scene_info.imheight
        imwidth = scene_info.imwidth

        active_bboxes = set()
        with TrackerInferenceReader(tracker_output_active, scene_name, image_height=imheight, image_width=imwidth) as tracker_inf_reader:
            for _ in tqdm(range(scene_length), desc=f'Preprocessing "{scene_name}"', unit='frame'):
                read = tracker_inf_reader.read()
                if read is None:
                    continue

                for tracklet_id in read.objects.keys():
                    active_bboxes.add((tracklet_id, read.frame_index))

        scene_video_path = os.path.join(tracker_output_option, f'{scene_name}.mp4')
        with TrackerInferenceReader(tracker_output_option, scene_name, image_height=imheight, image_width=imwidth) as tracker_inf_reader, \
            MP4Writer(scene_video_path, fps=cfg.visualize.fps) as mp4_writer:
            last_read = tracker_inf_reader.read()
            tracklet_presence_counter = Counter()  # Used to visualize new, appearing tracklets

            for index in tqdm(range(scene_length), desc=f'Visualizing "{scene_name}"', unit='frame'):
                image_path = dataset.get_scene_image_path(scene_name, index)
                frame = cv2.imread(image_path)
                assert frame is not None, \
                    f'Failed to load image for frame {index} on scene "{scene_name}" with path "{image_path}"!'

                if last_read is not None and index == last_read.frame_index:
                    for tracklet_id, bbox in last_read.objects.items():
                        tracklet_presence_counter[tracklet_id] += 1
                        draw_tracklet(
                            frame=frame,
                            tracklet_id=tracklet_id,
                            tracklet_age=tracklet_presence_counter[tracklet_id],
                            bbox=bbox,
                            new=tracklet_presence_counter[tracklet_id] <= cfg.visualize.new_object_length,
                            active=(tracklet_id, index) in active_bboxes
                        )

                    last_read = tracker_inf_reader.read()

                mp4_writer.write(frame)

@hydra.main(config_path=CONFIGS_PATH, config_name='movesort', version_base='1.1')
def main(cfg: DictConfig):
    # noinspection PyTypeChecker
    visualize_inference(cfg)


if __name__ == '__main__':
    main()
