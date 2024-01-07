"""
Momentum based association.
"""
from typing import Optional, List, Tuple

import numpy as np

from motrack.library.cv.bbox import PredBBox, Point
from motrack.tracker.matching.algorithms.base import AssociationAlgorithm
from motrack.tracker.matching.algorithms.utils import filter_observations
from motrack.tracker.matching.catalog import ASSOCIATION_CATALOG
from motrack.tracker.tracklet import Tracklet


class MomentumAssociation(AssociationAlgorithm):
    """
    Standard negative IoU association with gating.
    """
    def __init__(
        self,
        momentum: int = 1,
        fast_linear_assignment: bool = False
    ):
        """
        Args:
            momentum: Number of latest boxes (history) to consider for momentum association
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        super().__init__(fast_linear_assignment=fast_linear_assignment)

        self._momentum = momentum

    def form_cost_matrix(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        object_features: Optional[np.ndarray] = None,
        tracklets: Optional[List[Tracklet]] = None
    ) -> np.ndarray:
        _ = object_features  # Unused

        n_tracklets, n_detections = len(tracklet_estimations), len(detections)
        cost_matrix = np.zeros(shape=(n_tracklets, n_detections), dtype=np.float32)
        for t_i in range(n_tracklets):
            tracklet = tracklets[t_i]
            tracklet_bbox = tracklet_estimations[t_i]

            for d_i in range(n_detections):
                det_bbox = detections[d_i]

                cost_matrix[t_i][d_i] = self._calc_momentum(tracklet, tracklet_bbox, det_bbox, self._momentum)

        return cost_matrix

    @staticmethod
    def _calc_direction(point: Point, reference_point: Point) -> Tuple[float, float]:
        """
        Calculates unit direction given the point and the reference point.

        Direction: reference_point -> point

        Args:
            point: Point
            reference_point: Reference point

        Returns:
            Unit vector direction
        """
        direction_yx = point.y - reference_point.y, point.x - reference_point.x
        norm = np.sqrt(direction_yx[0] ** 2 + direction_yx[1] ** 2) + 1e-6
        direction_yx = (direction_yx[0] / norm, direction_yx[1] / norm)
        return direction_yx

    @staticmethod
    def momentum_score(
        tracklet_point: Point,
        tracklet_reference_point: Point,
        det_point: Point,
        det_reference_point: Point,
        conf: float
    ) -> float:
        """
        Calculates momentum score based on the tracklet direction and the candidate detection direction.

        Args:
            tracklet_point: Tracklet last point
            tracklet_reference_point: Tracklet previous point
            det_point: Detection point
            det_reference_point: Detection previous point
            conf: Detection confidence

        Returns:
            Momentum (angle) score
        """
        theta_inertia_y, theta_inertia_x = MomentumAssociation._calc_direction(tracklet_point, tracklet_reference_point)
        theta_intention_y, theta_intention_x = MomentumAssociation._calc_direction(det_point, det_reference_point)

        # scalar product: <x, y>
        # cos(theta) = <x, y> / (|x| * |y|)
        # |x| = |y| = 1
        # => theta = arcos(<x, y>)
        diff_angle_cos = theta_inertia_y * theta_intention_y + theta_inertia_x * theta_intention_x
        diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
        diff_angle = np.arccos(diff_angle_cos)
        diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi  # 1 in case of perfect match, 0 in case of bad match
        return - conf * diff_angle

    def _calc_momentum(self, tracklet: Tracklet, tracklet_bbox: PredBBox, det_bbox: PredBBox, momentum: int) -> float:
        """
        Calculates matching score (cost) between tracklet and a detection bounding box.

        Args:
            tracklet: Tracklet info
            tracklet_bbox: Tracklet bounding box
            det_bbox: Detection bounding box
            momentum: Momentum

        Returns:
            Momentum score
        """
        raise NotImplementedError('Score calculation is not implemented!')

@ASSOCIATION_CATALOG.register('ocm')
class OCM(MomentumAssociation):
    """
    Observation Centric Momentum. Reference: https://arxiv.org/pdf/2203.14360.pdf
    """
    def __init__(
        self,
        delta_t: int = 3,
        fast_linear_assignment: bool = False,
    ):
        """
        Args:
            delta_t: Time step between tracker bounding boxes to estimate the speed vector
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        super().__init__(fast_linear_assignment=fast_linear_assignment, momentum=1)
        self._delta_t = delta_t

    def _calc_momentum(self, tracklet: Tracklet, tracklet_bbox: PredBBox, det_bbox: PredBBox, momentum: int) -> float:
        observation_history = filter_observations(tracklet.history)
        current_frame_index = tracklet.frame_index + tracklet.lost_time + 1
        latest_observation_history = [(index, bbox) for index, bbox in observation_history if index >= current_frame_index - self._delta_t - 1]

        if len(latest_observation_history) < 2:
            if len(observation_history) < 2:
                return 0.0

            _, last_bbox = observation_history[-1]
            _, prev_bbox = observation_history[-2]
            _, after_prev_bbox = observation_history[-1]
        else:
            _, last_bbox = latest_observation_history[-1]
            _, prev_bbox = latest_observation_history[0]
            _, after_prev_bbox = latest_observation_history[1]

        return self.momentum_score(last_bbox.center, prev_bbox.center, det_bbox.center, after_prev_bbox.center, det_bbox.conf)


@ASSOCIATION_CATALOG.register('robust-ocm')
class RobustOCM(MomentumAssociation):
    """
    Robust OCM. Reference: https://arxiv.org/pdf/2308.00783.pdf
    """
    def __init__(
        self,
        momentum: int = 3,
        fast_linear_assignment: bool = False,
    ):
        """
        Args:
            momentum: Number of latest boxes (history) to consider for momentum association
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        super().__init__(fast_linear_assignment=fast_linear_assignment, momentum=momentum)

    @staticmethod
    def _get_corner_points(bbox: PredBBox) -> List[Point]:
        """
        Gets all bounding box corner points.

        Args:
            bbox: Bounding box

        Returns:
            Bounding box 4 corner points
        """
        return [
            Point(bbox.upper_left.x, bbox.upper_left.y),  # Upper Left
            Point(bbox.bottom_right.x, bbox.upper_left.y),  # Upper right
            Point(bbox.bottom_right.x, bbox.bottom_right.y),  # Bottom right
            Point(bbox.upper_left.x, bbox.bottom_right.y)  # Bottom Left
        ]

    def _calc_momentum(self, tracklet: Tracklet, tracklet_bbox: PredBBox, det_bbox: PredBBox, momentum: int) -> float:
        observation_history = filter_observations(tracklet.history)
        current_frame_index = tracklet.frame_index + tracklet.lost_time + 1
        latest_observation_history = [(index, bbox) for index, bbox in observation_history if index >= current_frame_index - self._momentum - 1]

        if len(latest_observation_history) < 2:
            if len(observation_history) < 2:
                return 0.0

            _, curr_bbox = observation_history[-1]
            _, prev_bbox = observation_history[-2]
            _, after_prev_bbox = observation_history[-1]

            momentum_score: float = 0.0
            for curr_point, prev_point, det_point, after_prev_point in zip(self._get_corner_points(curr_bbox), self._get_corner_points(prev_bbox),
                                                         self._get_corner_points(det_bbox), self._get_corner_points(after_prev_bbox)):
                momentum_score += self.momentum_score(curr_point, prev_point, det_point, after_prev_point, det_bbox.conf)

            return momentum_score / 4

        momentum_score: float = 0.0
        for i in range(1, self._momentum + 1):
            if i >= len(latest_observation_history):
                continue

            _, curr_bbox = latest_observation_history[-1]
            _, prev_bbox = latest_observation_history[-i - 1]
            _, after_prev_bbox = latest_observation_history[-i]

            for curr_point, prev_point, det_point, after_prev_point in zip(self._get_corner_points(curr_bbox), self._get_corner_points(prev_bbox),
                                                         self._get_corner_points(det_bbox), self._get_corner_points(after_prev_bbox)):
                momentum_score += self.momentum_score(curr_point, prev_point, det_point, after_prev_point, det_bbox.conf)

        return momentum_score / 4


@ASSOCIATION_CATALOG.register('ecm')
class ECM(OCM):
    """
    Estimation Centric Momentum. Like OCM but uses estimation instead of the last observation.
    """
    def _calc_momentum(self, tracklet: Tracklet, tracklet_bbox: PredBBox, det_bbox: PredBBox, momentum: int) -> float:
        if len(tracklet.history) < self._delta_t:
            prev_bbox = tracklet.bbox
        else:
            _, prev_bbox = tracklet.history[-self._delta_t]
        return self.momentum_score(tracklet_bbox.center, prev_bbox.center, det_bbox.center, prev_bbox.center, det_bbox.conf)


@ASSOCIATION_CATALOG.register('robust-ecm')
class RobustECM(RobustOCM):
    """
    Like Robust OCM but uses estimation instead of the last observation.
    """
    def _calc_momentum(self, tracklet: Tracklet, tracklet_bbox: PredBBox, det_bbox: PredBBox, momentum: int) -> float:
        if len(tracklet.history) < momentum:
            prev_bbox = tracklet.bbox

            momentum_score: float = 0.0
            for tracklet_point, prev_point, det_point in zip(self._get_corner_points(tracklet_bbox), self._get_corner_points(prev_bbox),
                                                             self._get_corner_points(det_bbox)):

                momentum_score += self.momentum_score(tracklet_point, prev_point, det_point, prev_point, det_bbox.conf)

            return momentum_score / 4

        momentum_score: float = 0.0
        for i in range(1, self._momentum + 1):
            _, prev_bbox = tracklet.history[-i]

            for tracklet_point, prev_point, det_point in zip(self._get_corner_points(tracklet_bbox), self._get_corner_points(prev_bbox),
                                             self._get_corner_points(det_bbox)):
                momentum_score += self.momentum_score(tracklet_point, prev_point, det_point, prev_point, det_bbox.conf)

        return momentum_score / 4
