"""
Height and bottom position distance
"""
from typing import Optional, List

import numpy as np

from motrack.library.cv.bbox import PredBBox
from motrack.tracker.matching.algorithms.base import AssociationAlgorithm
from motrack.tracker.matching.catalog import ASSOCIATION_CATALOG
from motrack.tracker.tracklet import Tracklet


@ASSOCIATION_CATALOG.register('hvc')
class HorizontalViewCues(AssociationAlgorithm):
    """
    Heuristic: When camera is positioned horizontally, bottom y position (bottom)
    and height become important cues.
    """
    def __init__(
        self,
        bottom_position_factor: float = 1.0,
        height_factor: float = 1.0,
        fast_linear_assignment: bool = False
    ):
        """
        Args:
            bottom_position_factor: Bottom position factor
            height_factor: Height factor
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        super().__init__(fast_linear_assignment=fast_linear_assignment)
        self._bottom_position_factor = bottom_position_factor
        self._height_factor = height_factor

    def form_cost_matrix(
            self,
            tracklet_estimations: List[PredBBox],
            detections: List[PredBBox],
            object_features: Optional[np.ndarray] = None,
            tracklets: Optional[List[Tracklet]] = None
    ) -> np.ndarray:
        _, _ = object_features, tracklets  # Unused

        n_tracklets, n_detections = len(tracklet_estimations), len(detections)
        cost_matrix = np.zeros(shape=(n_tracklets, n_detections), dtype=np.float32)
        for t_i in range(n_tracklets):
            tracklet_bbox = tracklet_estimations[t_i]

            for d_i in range(n_detections):
                det_bbox = detections[d_i]

                bottom_diff = abs(tracklet_bbox.bottom_right.y - det_bbox.bottom_right.y)
                height_diff = abs(tracklet_bbox.height - det_bbox.height)

                cost_matrix[t_i][d_i] = self._bottom_position_factor * bottom_diff + self._height_factor * height_diff

        return cost_matrix
