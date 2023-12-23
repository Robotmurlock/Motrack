"""
Implementation of Filter Sort tracker
"""
from typing import Tuple, Optional, List

import numpy as np
from abc import ABC, abstractmethod
import copy

from motrack.filter import filter_factory
from motrack.library.cv.bbox import PredBBox, BBox
from motrack.tracker.trackers.algorithms.base import Tracker
from motrack.tracker.tracklet import Tracklet, TrackletState, TrackletCommonData
from motrack.cmc import cmc_factory
from motrack.reid import reid_inference_factory


class MotionReIDBasedTracker(Tracker, ABC):
    """
    Baseline Sort tracker components:
    - ReID: HungarianIOU
    - Motion model: Filter prior
    - Combining detection and motion model: Filter posterior
    """
    def __init__(
        self,
        filter_name: str,
        filter_params: dict,
        cmc_name: Optional[str] = None,
        cmc_params: Optional[dict] = None,
        reid_name: Optional[str] = None,
        reid_params: Optional[str] = None,
        new_tracklet_detection_threshold: Optional[float] = None,
    ):
        """
        Args:
            filter_name: Filter name
            filter_params: Filter params
            cmc_name: CMC name
            cmc_params: CMC params
            reid_name: ReID name
            reid_params: ReID params
        """
        super().__init__()

        self._filter = filter_factory(filter_name, filter_params)

        self._cmc = None
        if cmc_name is not None:
            cmc_params = {} if cmc_params is None else cmc_params
            self._cmc = cmc_factory(cmc_name, cmc_params)

        self._reid = None
        if reid_name is not None:
            reid_params = {} if reid_params is None else reid_params
            self._reid = reid_inference_factory(reid_name, reid_params)

        # Hyperparameters
        self._new_tracklet_detection_threshold = new_tracklet_detection_threshold

        # State
        self._filter_states = {}
        self._next_id = 0

    @staticmethod
    def _raw_to_bbox(tracklet: Tracklet, raw: np.ndarray, conf: Optional[float] = None) -> PredBBox:
        """
        Converts raw tensor to PredBBox for tracked object.

        Args:
            tracklet: Tracklet
            raw: raw bbox coords
            conf: BBox confidence

        Returns:
            PredBBox
        """
        bbox_raw = raw.tolist()
        return PredBBox.create(
            bbox=BBox.from_xywh(*bbox_raw, clip=False),
            label=tracklet.bbox.label,
            conf=tracklet.bbox.conf if conf is None else conf
        )

    def _initiate(self, tracklet_id: int, detection: PredBBox) -> None:
        """
        Initiates new tracking object state.

        Args:
            tracklet_id: Tracklet id
            detection: Initial object detection
        """
        measurement = detection.as_numpy_xywh(dtype=np.float32)
        state = self._filter.initiate(measurement)
        self._filter_states[tracklet_id] = state

    def _predict(self, tracklet: Tracklet) -> Tuple[PredBBox, np.ndarray, np.ndarray]:
        """
        Estimates object prior position

        Args:
            tracklet: Tracklet (required to fetch state, and bbox info)

        Returns:
            Motion model prior estimation
        """
        assert tracklet.id in self._filter_states, f'Tracklet id "{tracklet.id}" can\'t be found in filter states!'
        state = self._filter_states[tracklet.id]
        state = self._filter.predict(state)
        prior_mean, prior_std = self._filter.project(state)
        self._filter_states[tracklet.id] = state
        bbox = self._raw_to_bbox(tracklet, prior_mean)
        return bbox, prior_mean, prior_std

    def _update(self, tracklet: Tracklet, detection: PredBBox) -> Tuple[PredBBox, np.ndarray, np.ndarray]:
        """
        Estimates object posterior position based on the matched detection.

        Args:
            tracklet: Tracklet (required to fetch state, and bbox info)

        Returns:
            Motion model posterior estimation
        """
        measurement = detection.as_numpy_xywh()

        state = self._filter_states[tracklet.id]
        state = self._filter.update(state, measurement)
        posterior_mean, posterior_std = self._filter.project(state)
        self._filter_states[tracklet.id] = state
        bbox = self._raw_to_bbox(tracklet, posterior_mean, conf=detection.conf)
        return bbox, posterior_mean, posterior_std

    def _missing(self, tracklet: Tracklet) -> Tuple[PredBBox, np.ndarray, np.ndarray]:
        """
        Estimates object posterior position for unmatched tracklets (with any detections)

        Args:
            tracklet: Tracklet (required to fetch state, and bbox info)

        Returns:
            Motion model posterior (missing) estimation
        """
        state = self._filter_states[tracklet.id]
        state = self._filter.missing(state)
        posterior_mean, posterior_std = self._filter.project(state)
        self._filter_states[tracklet.id] = state
        bbox = self._raw_to_bbox(tracklet, posterior_mean)
        return bbox, posterior_mean, posterior_std

    def _delete(self, tracklet_id: int) -> None:
        """
        Deletes filter state for deleted tracklet id.

        Args:
            tracklet_id: Tracklet id
        """
        self._filter_states.pop(tracklet_id)

    def _create_new_tracklets(
        self,
        detections: List[PredBBox],
        frame_index: int,
        objects_features: Optional[np.ndarray] = None,
        inplace: bool = True
    ) -> List[Tracklet]:
        new_tracklets: List[Tracklet] = []
        for d_i, detection in enumerate(detections):
            if self._new_tracklet_detection_threshold is not None and detection.conf < self._new_tracklet_detection_threshold:
                continue

            new_tracklet = Tracklet(
                bbox=detection if inplace else copy.deepcopy(detection),
                frame_index=frame_index,
                _id=self._next_id,
                state=TrackletState.NEW if frame_index > 1 else TrackletState.ACTIVE
            )
            self._next_id += 1
            new_tracklets.append(new_tracklet)
            self._initiate(new_tracklet.id, detection)

            # Set initial tracklet feature if ReId model is used
            if objects_features is not None:
                new_tracklet.set(TrackletCommonData.APPEARANCE, objects_features[d_i])

        return new_tracklets

    def _perform_cmc(self, frame: np.ndarray, frame_index: int, bboxes: List[PredBBox]) -> List[PredBBox]:
        if self._cmc is None:
            return bboxes

        warp = self._cmc.apply(frame, frame_index, scene=self._scene)
        self._filter_states = {t_id: self._filter.affine_transform(state, warp) for t_id, state in self._filter_states.items()}

        corrected_bboxes: List[PredBBox] = []
        for bbox in bboxes:
            bbox = PredBBox.create(
                bbox=bbox.affine_transform(warp),
                label=bbox.label,
                conf=bbox.conf
            )
            corrected_bboxes.append(bbox)

        return corrected_bboxes


    def _extract_reid_features(self, frame: np.ndarray, frame_index: int, bboxes: List[PredBBox]) -> Optional[np.ndarray]:
        if self._reid is None:
            return None

        return self._reid.extract_objects_features(frame, bboxes, frame_index=frame_index, scene=self._scene)

    def track(
        self,
        tracklets: List[Tracklet],
        detections: List[PredBBox],
        frame_index: int,
        inplace: bool = True,
        frame: Optional[np.ndarray] = None
    ) -> List[Tracklet]:
        frame_index -= 1
        tracklets, prior_tracklet_bboxes = self._track_predict(tracklets, frame_index, frame=frame)
        objects_features = self._extract_reid_features(frame, frame_index, detections)
        return self._track(tracklets, prior_tracklet_bboxes, detections, frame_index,
                           objects_features=objects_features, inplace=inplace, frame=frame)

    def _track_predict(self, tracklets: List[Tracklet], frame_index: int, frame: Optional[np.ndarray] = None):
        """
        Performs tracker tracklet filtering motion prediction steps:
        - Removes DELETED trackelts
        - Uses filter to obtain prior tracklet position estimates
        - Optionally performs CMC for prediction corrections

        Args:
            tracklets: List of tracklets
            frame_index: Current frame index
            frame: Current frame image (optional)

        Returns:
            - Filtered tracklets
            - Tracklet object position predictions for the current frame
        """
        tracklets = [t for t in tracklets if t.state != TrackletState.DELETED]  # Remove deleted tracklets

        # Estimate priors for all tracklets
        prior_tracklet_estimates = [self._predict(t) for t in tracklets]
        prior_tracklet_bboxes = [bbox for bbox, _, _ in prior_tracklet_estimates]
        prior_tracklet_bboxes = self._perform_cmc(frame, frame_index, prior_tracklet_bboxes)

        return tracklets, prior_tracklet_bboxes

    @abstractmethod
    def _track(
        self,
        tracklets: List[Tracklet],
        prior_tracklet_bboxes: List[PredBBox],
        detections: List[PredBBox],
        frame_index: int,
        objects_features: Optional[np.ndarray] = None,
        inplace: bool = True,
        frame: Optional[np.ndarray] = None
    ) -> List[Tracklet]:
        """
        Main tracker logic (after prior estimates and CMC).

        Args:
            tracklets: List of active trackelts
            prior_tracklet_bboxes: Tracklet prior position bboxes
            detections: Lists of new detections
            frame_index: Current frame number
            objects_features: Object appearance features
            inplace: Perform inplace transformations on tracklets and bboxes
            frame: Pass frame in case CMC or appearance based association is used

        Returns:
            Tracks (active, lost, new, deleted, ...)
        """
