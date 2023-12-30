"""
Visualization tools for:
- Tracker inference
- Object detection inference
"""
import os
import re
from collections import Counter
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from motrack.config_parser import TrackerVisualizeConfig
from motrack.datasets import BaseDataset
from motrack.evaluation.io import TrackerInferenceReader
from motrack.library.cv import color_palette
from motrack.library.cv.bbox import PredBBox, BBox
from motrack.library.cv.drawing import draw_text
from motrack.library.cv.video_writer import MP4Writer
from motrack.object_detection import DetectionManager


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
    text = [f'id={tracklet_id}', f'age={tracklet_age}', f'conf={100 * bbox.conf:.0f}%']
    return draw_text(frame, text, left, top, color=color)


def run_visualize_tracker_inference(
    dataset: BaseDataset,
    tracker_active_output: str,
    tracker_output_option: str,
    visualize_cfg: Optional[TrackerVisualizeConfig] = None,
    scene_pattern: str = '(.*?)'
) -> None:
    """
    Visualizes tracker inference output.

    Args:
        dataset: Dataset
        tracker_active_output: Path where tracker ACTIVE tracks inference can be found
        tracker_output_option: Path to any type of tracker inference to visualize (active, all, or postprocess)
            Recommended: All
        visualize_cfg: Custom visualization config (if not set then default is used)
        scene_pattern: Scene pattern filter (regex)

    Returns:

    """
    if visualize_cfg is None:
        visualize_cfg = TrackerVisualizeConfig()

    scene_names = dataset.scenes
    scene_names = [scene_name for scene_name in scene_names if re.match(scene_pattern, scene_name)]
    for scene_name in tqdm(scene_names, desc='Visualizing tracker', unit='scene'):
        scene_info = dataset.get_scene_info(scene_name)
        scene_length = scene_info.seqlength
        imheight = scene_info.imheight
        imwidth = scene_info.imwidth

        active_bboxes = set()
        with TrackerInferenceReader(tracker_active_output, scene_name, image_height=imheight, image_width=imwidth) as tracker_inf_reader:
            for _ in tqdm(range(scene_length), desc=f'Preprocessing "{scene_name}"', unit='frame'):
                read = tracker_inf_reader.read()
                if read is None:
                    continue

                for tracklet_id in read.objects.keys():
                    active_bboxes.add((tracklet_id, read.frame_index))

        scene_video_path = os.path.join(tracker_output_option, f'{scene_name}.mp4')
        with TrackerInferenceReader(tracker_output_option, scene_name, image_height=imheight, image_width=imwidth) as tracker_inf_reader, \
                MP4Writer(scene_video_path, fps=visualize_cfg.fps) as mp4_writer:
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
                            new=tracklet_presence_counter[tracklet_id] <= visualize_cfg.new_object_length,
                            active=(tracklet_id, index) in active_bboxes
                        )

                    last_read = tracker_inf_reader.read()

                mp4_writer.write(frame)

def run_visualize_detections(
    dataset: BaseDataset,
    detection_manager: DetectionManager,
    output_path: str,
    fps: int = 30
) -> None:
    """
    Performs visualization of raw detections without usage of the tracker.

    Args:
        dataset: Dataset to perform tracker inference on
        detection_manager: Detection manager
        output_path: Path where the detections will be stored
        fps: output video fps
    """
    scene_names = dataset.scenes
    for scene_name in tqdm(scene_names, desc='Visualize detections', unit='scene'):
        scene_info = dataset.get_scene_info(scene_name)
        scene_length = scene_info.seqlength
        video_path = os.path.join(output_path, f'{scene_name}.mp4')

        with MP4Writer(video_path, fps=fps) as video_writer:
            for index in tqdm(range(scene_length), desc=f'Visualizing "{scene_name}"', unit='frame'):
                image = dataset.load_scene_image_by_frame_index(scene_name, index)
                detection_bboxes = detection_manager.predict(scene_name, index)
                for bbox in detection_bboxes:
                    image = bbox.draw(image)

                video_writer.write(image)
