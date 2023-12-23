"""
TrackEval inference IO support.
"""
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, Dict, Optional

from motrack.library.cv.bbox import PredBBox, BBox
from motrack.tracker.tracklet import Tracklet

TRACKER_INFERENCE_HEADER = [
    'frame_id', 'id',
    'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'conf',
    '-1_0', '-1_1', '-1_2'
]


class TrackerInferenceWriter:
    """
    Writes tracker inference in MOT format. Can be used afterward for evaluation.
    (RAII)
    """
    def __init__(self, output_path: str, scene_name: str, image_height: int, image_width: int, clip: bool = True):
        """
        Args:
            output_path: Tracker inference output directory path
            scene_name: Scene name
            image_height: Image height (required for coordinates up-scaling)
            image_width: Image height (required for coordinates up-scaling)
            clip: Clip tracklet bbox
        """
        self._output_path = output_path
        self._scene_name = scene_name
        self._scene_output_path = os.path.join(output_path, f'{scene_name}.txt')
        self._image_height = image_height
        self._image_width = image_width
        self._clip = clip

        # State
        self._writer = None

    def open(self) -> None:
        """
        Opens writer.
        """
        Path(self._output_path).mkdir(parents=True, exist_ok=True)
        self._writer: TextIO = open(self._scene_output_path, 'w', encoding='utf-8')  # pylint: disable=consider-using-with

    def close(self) -> None:
        """
        Closes writer.
        """
        if self._writer is not None:
            self._writer.close()

    def write(self, frame_index: int, tracklet: Tracklet) -> None:
        """
        Writes info about one tracker tracklet.
        One tracklet - one row.

        Args:
            frame_index: Current frame index
            tracklet: Tracklet
        """
        bbox = tracklet.bbox
        if self._clip:
            bbox = copy.deepcopy(bbox)
            bbox.clip()

        left = round(bbox.upper_left.x * self._image_width)
        top = round(bbox.upper_left.y * self._image_height)
        width = round(bbox.width * self._image_width)
        height = round(bbox.height * self._image_height)

        cells = [
            str(frame_index + 1), str(tracklet.id),
            str(left), str(top),
            str(width), str(height),
            str(bbox.conf),
            '-1', '-1', '-1'
        ]

        row = ','.join(cells)
        self._writer.write(f'{row}\n')

    def __enter__(self) -> 'TrackerInferenceWriter':
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


@dataclass
class TrackerInferenceRead:
    frame_index: int
    frame_id: str
    objects: Dict[str, PredBBox]


class TrackerInferenceReader:
    """
    Reads tracker inference in MOT format.
    (RAII)
    """
    BBOX_LABEL = 'Person'

    def __init__(self, output_path: str, scene_name: str, image_height: int, image_width: int):
        """
        Args:
            output_path: Tracker inference output directory path
            scene_name: Scene name
            image_height: Image height (required for coordinates up-scaling)
            image_width: Image height (required for coordinates up-scaling)
        """
        self._output_path = output_path
        self._scene_name = scene_name
        self._scene_output_path = os.path.join(output_path, f'{scene_name}.txt')
        self._image_height = image_height
        self._image_width = image_width

        # State
        self._reader = None
        self._curr_frame_id = None
        self._last_line = None

    def open(self) -> None:
        """
        Opens reader.
        """
        self._reader: TextIO = open(self._scene_output_path, 'r', encoding='utf-8')  # pylint: disable=consider-using-with

    def close(self) -> None:
        """
        Closes reader.
        """
        if self._reader is not None:
            self._reader.close()

    def read(self) -> Optional[TrackerInferenceRead]:
        """
        Reads information about next frame.

        Returns:
            Dictionary with fields:
            - `frame_index` for frame index
            - `frame_id` for frame id (frame_index + 1)
            - `objects` all objects present in the current frame
        """
        line = self._reader.readline() if self._last_line is None else self._last_line
        if not line:
            # End of file
            return None
        self._last_line = None

        data = None
        while True:
            frame_id, tracklet_id, bbox_left, bbox_top, bbox_width, bbox_height, conf, _, _, _ = line.split(',')
            frame_index = int(frame_id) - 1
            if self._curr_frame_id is None or self._curr_frame_id == frame_id:
                if self._curr_frame_id is None:
                    # Create data object
                    data = TrackerInferenceRead(
                        frame_index=frame_index,
                        frame_id=frame_id,
                        objects={}
                    )

                    self._curr_frame_id = frame_id

                # Save object
                bbox_left, bbox_top, bbox_width, bbox_height, conf = [float(v) for v in [bbox_left, bbox_top, bbox_width, bbox_height, conf]]
                bbox = PredBBox.create(
                    bbox=BBox.from_xywh(
                        bbox_left / self._image_width,
                        bbox_top / self._image_height,
                        bbox_width / self._image_width,
                        bbox_height / self._image_height
                    ),
                    label=self.BBOX_LABEL,  # Note: This is specialized for MOT20 format
                    conf=conf
                )
                data.objects[tracklet_id] = bbox

                # Read next line
                line = self._reader.readline()
                if not line:
                    # End of file
                    self._curr_frame_id = None
                    self._last_line = None
                    return data
            else:
                self._curr_frame_id = None
                self._last_line = line
                return data

    def __enter__(self) -> 'TrackerInferenceReader':
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
