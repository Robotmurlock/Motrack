"""
Tracker inference tool.
"""
import dataclasses
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional

from tqdm import tqdm
import yaml

from motrack.common import conventions
from motrack.config_parser import GlobalConfig
from motrack.datasets import BaseDataset
from motrack.inference.io import TrackerInferenceWriter
from motrack.object_detection import DetectionManager
from motrack.tools.dataset_builder import DatasetBuilder, default_dataset_builder
from motrack.tracker import Tracker, tracker_factory
from motrack.tracker.tracklet import Tracklet, TrackletState

logger = logging.getLogger('TrackerInference')


@dataclass
class OptunaOutputData:
    """Optuna trial metadata attached to a tracker run."""
    study_name: str
    trial_number: int
    trial_params: Dict[str, Any]


@dataclass
class InferenceOutputData:
    """Metadata for a single tracker run (``run_meta.json``)."""
    created_at: str
    optuna: Optional[OptunaOutputData] = None

    def to_dict(self) -> dict:
        d: dict = {'created_at': self.created_at}
        if self.optuna is not None:
            d['optuna'] = dataclasses.asdict(self.optuna)
        return d

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'InferenceOutputData':
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        optuna_raw = raw.pop('optuna', None)
        optuna = OptunaOutputData(**optuna_raw) if optuna_raw is not None else None
        return cls(created_at=raw['created_at'], optuna=optuna)


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


def run_inference(
    cfg: GlobalConfig,
    inference_output: Optional[InferenceOutputData] = None,
    dataset_builder: DatasetBuilder = default_dataset_builder,
) -> None:
    """High-level inference orchestration.

    Builds dataset/detector/tracker, runs the per-frame loop via
    ``run_tracker_inference``, saves the config snapshot and run metadata,
    and optionally postprocesses the output.

    Args:
        cfg: Global config.
        inference_output: Optional pre-populated metadata; if None, a fresh
            ``InferenceOutputData`` is created. Used by the optimizer to
            attach Optuna trial info.
        dataset_builder: Pluggable dataset construction. Defaults to
            ``motrack.datasets.dataset_factory``.
    """
    # Lazy import to avoid circular dependency:
    # motrack.tools.__init__ → optimization → inference. The
    # postprocess module is independent and safe to import here.
    from motrack.tools.postprocess import run_tracker_postprocess

    if os.path.exists(cfg.experiment_path):
        if cfg.inference.override:
            user_input = 'yes'
        else:
            user_input = input(f'Experiment on path "{cfg.experiment_path}" already exists. '
                               f'Are you sure you want to override it? [yes/no] ').lower()
        if user_input in ['yes', 'y']:
            shutil.rmtree(cfg.experiment_path)
        else:
            logger.info('Aborting!')
            return

    tracker_online_output = conventions.get_tracker_output_path(
        cfg.experiment_path,
        conventions.TrackerOutputType.ONLINE,
    )
    tracker_debug_output = conventions.get_tracker_output_path(
        cfg.experiment_path,
        conventions.TrackerOutputType.DEBUG,
    )

    logger.info(f'Saving tracker inference on path "{cfg.experiment_path}".')

    dataset = dataset_builder(cfg)

    detection_manager = DetectionManager(
        inference_name=cfg.object_detection.type,
        inference_params=cfg.object_detection.params,
        lookup=cfg.object_detection.load_lookup() if cfg.object_detection.lookup_path is not None else None,
        dataset=dataset,
        cache_path=cfg.object_detection.cache_path,
        oracle=cfg.object_detection.oracle,
    )

    tracker = tracker_factory(
        name=cfg.algorithm.name,
        params=cfg.algorithm.params,
    )

    fps_output_path = conventions.get_fps_stats_path(cfg.experiment_path)
    run_tracker_inference(
        dataset=dataset,
        tracker=tracker,
        detection_manager=detection_manager,
        tracker_active_output=tracker_online_output,
        tracker_all_output=tracker_debug_output,
        clip=cfg.inference.clip,
        scene_pattern=cfg.dataset_filter.scene_pattern,
        load_image=cfg.inference.load_image,
        fps_output_path=fps_output_path,
    )

    tracker_config_path = conventions.get_config_snapshot_path(cfg.experiment_path)
    with open(tracker_config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(asdict(cfg), f)

    inference_output_path = conventions.get_run_meta_path(cfg.experiment_path)
    if inference_output is None:
        inference_output = InferenceOutputData(created_at=datetime.now().isoformat())
    inference_output.save(inference_output_path)

    if cfg.inference.postprocess:
        logger.info('Performing inference postprocessing...')
        tracker_offline_output = conventions.get_tracker_output_path(
            cfg.experiment_path,
            conventions.TrackerOutputType.OFFLINE,
        )
        run_tracker_postprocess(
            dataset=dataset,
            tracker_active_output=tracker_online_output,
            tracker_all_output=tracker_debug_output,
            tracker_postprocess_output=tracker_offline_output,
            postprocess_cfg=cfg.postprocess,
            scene_pattern=cfg.dataset_filter.scene_pattern,
            clip=cfg.inference.clip,
        )
