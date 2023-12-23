"""
Tracker inference tool.
"""
import re
from typing import List

from tqdm import tqdm

from motrack.datasets import BaseDataset
from motrack.evaluation.io import TrackerInferenceWriter
from motrack.object_detection import DetectionManager
from motrack.tracker import Tracker
from motrack.tracker.tracklet import Tracklet, TrackletState


def run_tracker_inference(
    dataset: BaseDataset,
    tracker: Tracker,
    detection_manager: DetectionManager,
    tracker_active_output: str,
    tracker_all_output: str,
    clip: bool = True,
    scene_pattern: str = '(.*?)'
) -> None:
    """
    Performs inference on given dataset with a given tracker and detection manager.

    Args:
        dataset: Dataset to perform tracker inference on
        tracker: Tracker
        detection_manager: Detection manager
        tracker_active_output: Path where the active tracks are stored
        tracker_all_output: Path where the all tracks are stored
        clip: Clip bounding boxes coordinates to range [0, 1]
        scene_pattern: Filter dataset scenes.
    """
    scene_names = dataset.scenes
    scene_names = [scene_name for scene_name in scene_names if re.match(scene_pattern, scene_name)]
    for scene_name in tqdm(scene_names, desc='Simulating tracker', unit='scene'):
        tracker.reset_state()
        tracker.set_scene(scene_name)
        tracklets: List[Tracklet] = []

        scene_info = dataset.get_scene_info(scene_name)
        scene_length = scene_info.seqlength
        imheight = scene_info.imheight
        imwidth = scene_info.imwidth

        with TrackerInferenceWriter(tracker_active_output, scene_name, image_height=imheight, image_width=imwidth,
                                    clip=clip) as tracker_active_inf_writer, \
                TrackerInferenceWriter(tracker_all_output, scene_name, image_height=imheight, image_width=imwidth,
                                       clip=clip) as tracker_all_inf_writer:
            for index in tqdm(range(scene_length), desc=f'Simulating "{scene_name}"', unit='frame'):
                # Perform OD inference
                detection_bboxes = detection_manager.predict(scene_name, index)

                # Perform tracking step
                tracklets = tracker.track(
                    tracklets=tracklets,
                    detections=detection_bboxes,
                    frame_index=index + 1,  # Counts from 1 instead of 0
                    frame=dataset.load_scene_image_by_frame_index(scene_name, index)
                )
                active_tracklets = [t for t in tracklets if t.state == TrackletState.ACTIVE]

                # Save inference
                for tracklet in active_tracklets:
                    tracker_active_inf_writer.write(index, tracklet)

                for tracklet in tracklets:
                    tracker_all_inf_writer.write(index, tracklet)
