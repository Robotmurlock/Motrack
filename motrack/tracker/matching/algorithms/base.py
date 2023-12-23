"""
Association method interface.
"""
from typing import List, Optional, Tuple

import numpy as np

from motrack.library.cv.bbox import PredBBox
from motrack.tracker.matching.utils import hungarian, greedy
from motrack.tracker.tracklet import Tracklet


class AssociationAlgorithm:
    """
    Defines interface to tracklets-detections association matching.
    """
    def __init__(
        self,
        fast_linear_assignment: bool = False,
    ):
        """
        fast_linear_assignment: Use greedy algorithm for linear assignment
            - This might be more efficient in case of large cost matrix
        """
        self._fast_linear_assignment = fast_linear_assignment

    def form_cost_matrix(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        object_features: Optional[np.ndarray] = None,
        tracklets: Optional[List[Tracklet]] = None
    ) -> np.ndarray:
        """
        Forms matching cost matrix used as the input to the linear assignment solver.

        Args:
            tracklet_estimations: Tracked object from previous frames
            detections: Currently detected objects
            object_features: Object appearance features (optional)
            tracklets: Full tracklet info (optional)

        Returns:
            Cost matrix
        """
        raise NotImplementedError('Form cost matrix is not implemented!')

    def match(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        object_features: Optional[np.ndarray] = None,
        tracklets: Optional[List[Tracklet]] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Performs matching between tracklets and detections (interface).

        Args:
            tracklet_estimations: Tracked object from previous frames
            detections: Currently detected objects
            object_features: Object appearance features (optional)
            tracklets: Full tracklet info (optional)

        Returns:
            - List of matches (pairs)
            - list of unmatched tracklets
            - list of unmatched detections
        """
        cost_matrix = self.form_cost_matrix(
            tracklet_estimations=tracklet_estimations,
            detections=detections,
            object_features=object_features,
            tracklets=tracklets
        )

        return greedy(cost_matrix) if self._fast_linear_assignment else \
            hungarian(cost_matrix)

    def __call__(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        object_features: Optional[np.ndarray] = None,
        tracklets: Optional[List[Tracklet]] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        return self.match(tracklet_estimations, detections, object_features=object_features, tracklets=tracklets)
