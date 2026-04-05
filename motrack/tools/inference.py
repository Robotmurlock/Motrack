"""
Tracker inference tool.
"""
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from tqdm import tqdm

from motrack.datasets import BaseDataset
from motrack.inference.io import TrackerInferenceWriter
from motrack.object_detection import DetectionManager
from motrack.tracker import Tracker
from motrack.tracker.tracklet import Tracklet, TrackletState

logger = logging.getLogger('TrackerInference')


@dataclass
class SceneFPSStats:
    """FPS statistics for a single scene."""
    scene_name: str
    n_frames: int
    detection_total_s: float
    association_total_s: float
    e2e_total_s: float

    @property
    def detection_fps(self) -> float:
        return self.n_frames / max(self.detection_total_s, 1e-9)

    @property
    def association_fps(self) -> float:
        return self.n_frames / max(self.association_total_s, 1e-9)

    @property
    def e2e_fps(self) -> float:
        return self.n_frames / max(self.e2e_total_s, 1e-9)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['detection_fps'] = round(self.detection_fps, 2)
        d['association_fps'] = round(self.association_fps, 2)
        d['e2e_fps'] = round(self.e2e_fps, 2)
        return d


@dataclass
class InferenceFPSStats:
    """Aggregated FPS statistics across all scenes."""
    scenes: List[SceneFPSStats] = field(default_factory=list)

    @property
    def total_frames(self) -> int:
        return sum(s.n_frames for s in self.scenes)

    @property
    def detection_fps(self) -> float:
        total_frames = self.total_frames
        total_time = sum(s.detection_total_s for s in self.scenes)
        return total_frames / max(total_time, 1e-9)

    @property
    def association_fps(self) -> float:
        total_frames = self.total_frames
        total_time = sum(s.association_total_s for s in self.scenes)
        return total_frames / max(total_time, 1e-9)

    @property
    def e2e_fps(self) -> float:
        total_frames = self.total_frames
        total_time = sum(s.e2e_total_s for s in self.scenes)
        return total_frames / max(total_time, 1e-9)

    def to_dict(self) -> dict:
        return {
            'total_frames': self.total_frames,
            'detection_fps': round(self.detection_fps, 2),
            'association_fps': round(self.association_fps, 2),
            'e2e_fps': round(self.e2e_fps, 2),
            'scenes': [s.to_dict() for s in self.scenes],
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)


def run_tracker_inference(
    dataset: BaseDataset,
    tracker: Tracker,
    detection_manager: DetectionManager,
    tracker_active_output: str,
    tracker_all_output: str,
    clip: bool = True,
    scene_pattern: str = '(.*?)',
    load_image: bool = True,
    fps_output_path: Optional[str] = None,
) -> InferenceFPSStats:
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
        load_image: Load image for Object Detection or ReID model
            - Can be set to False if everything is already cached
        fps_output_path: If set, dump FPS statistics to this JSON path

    Returns:
        FPS statistics
    """
    fps_stats = InferenceFPSStats()

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

        detection_total_s = 0.0
        association_total_s = 0.0

        with TrackerInferenceWriter(tracker_active_output, scene_name, image_height=imheight, image_width=imwidth,
                                    clip=clip) as tracker_active_inf_writer, \
                TrackerInferenceWriter(tracker_all_output, scene_name, image_height=imheight, image_width=imwidth,
                                       clip=clip) as tracker_all_inf_writer:

            scene_start = time.perf_counter()

            for index in tqdm(range(scene_length), desc=f'Simulating "{scene_name}"', unit='frame'):
                # Perform OD inference
                t0 = time.perf_counter()
                detection_bboxes = detection_manager.predict(scene_name, index)
                detection_total_s += time.perf_counter() - t0

                # Perform tracking step
                t0 = time.perf_counter()
                tracklets = tracker.track(
                    tracklets=tracklets,
                    detections=detection_bboxes,
                    frame_index=index + 1,  # Counts from 1 instead of 0
                    frame=dataset.load_scene_image_by_frame_index(scene_name, index) if load_image else None
                )
                association_total_s += time.perf_counter() - t0

                active_tracklets = [t for t in tracklets if t.state == TrackletState.ACTIVE]

                # Save inference
                for tracklet in active_tracklets:
                    tracker_active_inf_writer.write(index, tracklet)

                for tracklet in tracklets:
                    tracker_all_inf_writer.write(index, tracklet)

            e2e_total_s = time.perf_counter() - scene_start

        scene_stats = SceneFPSStats(
            scene_name=scene_name,
            n_frames=scene_length,
            detection_total_s=detection_total_s,
            association_total_s=association_total_s,
            e2e_total_s=e2e_total_s,
        )
        fps_stats.scenes.append(scene_stats)
        logger.info(
            f'Scene "{scene_name}" ({scene_length} frames): '
            f'det={scene_stats.detection_fps:.1f} FPS, '
            f'assoc={scene_stats.association_fps:.1f} FPS, '
            f'e2e={scene_stats.e2e_fps:.1f} FPS'
        )

    logger.info(
        f'Total ({fps_stats.total_frames} frames): '
        f'det={fps_stats.detection_fps:.1f} FPS, '
        f'assoc={fps_stats.association_fps:.1f} FPS, '
        f'e2e={fps_stats.e2e_fps:.1f} FPS'
    )

    if fps_output_path is not None:
        fps_stats.save(fps_output_path)
        logger.info(f'FPS stats saved to "{fps_output_path}"')

    return fps_stats
