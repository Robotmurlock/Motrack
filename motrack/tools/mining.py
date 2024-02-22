"""
Mines detections based on IoU for motion filter training.
"""
import logging
import os
from pathlib import Path
from typing import List

from tqdm import tqdm

from motrack.datasets import BaseDataset
from motrack.library.cv.bbox import PredBBox, BBox
from motrack.object_detection import DetectionManager
from motrack.tracker.matching import IoUAssociation
from motrack.tracker.tracklet import Tracklet

logger = logging.getLogger('DetectionMining')


class DetectionWriter:
    """
    Writes detections in format same as the tracker ground truths.
    """
    def __init__(self, path: str):
        """
        Args:
            path: Output path
        """
        self._path = path

        # State
        self._writer = None

    def open(self) -> None:
        """
        Opens file writer.
        """
        self._writer = open(self._path, 'w', encoding='utf-8')  # pylint: disable=consider-using-with

    def close(self) -> None:
        """
        Closes file writer.
        """
        self._writer.close()

    def write(self, object_id: int, index: int, bbox: List[float]) -> None:
        """
        Writes detection info as a row.

        Args:
            object_id: Object id
            index: Frame index
            bbox: Bounding box (un-normalized coordinates)
        """
        cells = [
            index + 1, object_id,
            *bbox,
            '1', '1', '1'
        ]
        row = ','.join([str(c) for c in cells])
        self._writer.write(f'{row}\n')

    def __enter__(self) -> 'DetectionWriter':
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def run_detection_mining(
    dataset: BaseDataset,
    detection_manager: DetectionManager,
    output_path: str,
    iou_threshold: float = 0.7,
    min_confidence: float = 0.3
) -> None:
    """
    Performs visualization of raw detections without usage of the tracker.

    Args:
        dataset: Dataset to perform tracker inference on
        detection_manager: Detection manager
        output_path: Path where the mined detections are stored
        iou_threshold: IoU threshold to consider detection as a positive
        min_confidence: Minimum confidence
    """
    iou_assoc = IoUAssociation(match_threshold=iou_threshold)

    n_global_matches = 0
    n_global_gts = 0
    total_global_iou_score = 0.0

    scene_names = dataset.scenes
    for scene_name in tqdm(scene_names, desc='Mining detections', unit='scene'):
        scene_info = dataset.get_scene_info(scene_name)
        scene_length = scene_info.seqlength
        object_ids = dataset.get_scene_object_ids(scene_name)

        n_scene_matches = 0
        n_scene_gts = 0
        total_scene_iou_score = 0.0

        Path(output_path).mkdir(parents=True, exist_ok=True)
        with DetectionWriter(os.path.join(output_path, f'{scene_name}.txt')) as detection_writer:
            for index in tqdm(range(scene_length), desc=f'Mining detections on "{scene_name}"', unit='frame'):
                # Prepare detection bounding boxes
                detection_bboxes = detection_manager.predict(scene_name, index)
                detection_bboxes = [bbox for bbox in detection_bboxes if bbox.conf >= min_confidence]

                # Prepare GT bounding boxes
                gt_data = [(object_id, dataset.get_object_data_by_frame_index(object_id, index)) for object_id in object_ids]
                gt_data = [(object_id, PredBBox.create(BBox.from_xywh(*data.bbox), conf=1.0, label=data.category))
                           for object_id, data in gt_data if data is not None]
                gt_bboxes = [bbox for _, bbox in gt_data]

                gt_tracklets = [Tracklet(bbox, frame_index=index) for bbox in gt_bboxes]
                matches, _, _ = iou_assoc(gt_bboxes, detection_bboxes, tracklets=gt_tracklets)
                n_scene_matches += len(matches)
                total_scene_iou_score += sum(gt_bboxes[gt_i].iou(detection_bboxes[d_i]) for gt_i, d_i in matches)
                n_scene_gts += len(gt_bboxes)

                for gt_i, d_i in matches:
                    bbox = detection_bboxes[d_i].as_numpy_xywh()
                    bbox = [round(bbox[0] * scene_info.imwidth), round(bbox[1] * scene_info.imheight),
                            round(bbox[2] * scene_info.imwidth), round(bbox[3] * scene_info.imheight)]

                    object_id, _ = gt_data[gt_i]
                    _, scene_object_id = dataset.parse_object_id(object_id)
                    scene_object_id = int(scene_object_id)

                    detection_writer.write(scene_object_id, index, bbox)

        logger.info(f'Scene "{scene_name}" has {100 * n_scene_matches / n_scene_gts:.1f} mined ratio.')
        logger.info(f'Scene "{scene_name}" has {100 * total_scene_iou_score / n_scene_matches:.1f} average overlap!.')

        n_global_matches += n_scene_matches
        n_global_gts += n_scene_gts
        total_global_iou_score += total_scene_iou_score

    logger.info(f'Global {100 * n_global_matches / n_global_gts:.1f} mined ratio.')
    logger.info(f'Global {100 * total_global_iou_score / n_global_matches:.1f} average overlap!.')
    logger.info(f'Mined detections path: "{output_path}".')
