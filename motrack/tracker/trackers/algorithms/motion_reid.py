"""
Implementation of Filter Sort tracker
"""
from typing import Tuple, Optional, List, Union

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
        reid_detection_threshold: Optional[float] = None,
        initialization_threshold: int = 3,
        remember_threshold: int = 1,
        new_tracklet_detection_threshold: Optional[float] = None,
        use_observation_if_lost: bool = False,
        appearance_ema_momentum: float = 0.95
    ):
        """
        Args:
            filter_name: Filter name
            filter_params: Filter params
            cmc_name: CMC name
            cmc_params: CMC params
            reid_name: ReID name
            reid_params: ReID params

            reid_detection_threshold: Filter bboxes that have more than `reid_detection_threshold`
                detection score before applying ReID. Mandatory for trackers that use low detection scores
                like `BYTE`.

            remember_threshold: How long does the tracklet without any detection matching
                If tracklet isn't matched for more than `remember_threshold` frames then
                it is deleted.
            initialization_threshold: Number of frames until tracklet becomes active
            new_tracklet_detection_threshold: Threshold to accept new tracklet
            use_observation_if_lost: When re-finding tracklet, use observation instead of estimation
            appearance_ema_momentum: Appearance EMA (Exponential Moving Average) momentum
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
        self._reid_detection_threshold = reid_detection_threshold
        self._new_tracklet_detection_threshold = new_tracklet_detection_threshold
        self._initialization_threshold = initialization_threshold
        self._remember_threshold = remember_threshold
        self._use_observation_if_lost = use_observation_if_lost

        # ReID hyperparameters
        self._appearance_ema_momentum = appearance_ema_momentum

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
        objects_features: Optional[np.ndarray] = None
    ) -> List[Tracklet]:
        """
        Creates new tracklets based on unmatched detections.

        Args:
            detections: List of unmatched detections
            frame_index: Current frame index
            objects_features: List of unmatched detections' object features

        Returns:
            New tracklets
        """
        new_tracklets: List[Tracklet] = []
        for d_i, detection in enumerate(detections):
            if self._new_tracklet_detection_threshold is not None and detection.conf < self._new_tracklet_detection_threshold:
                continue

            new_tracklet = Tracklet(
                bbox=copy.deepcopy(detection),
                frame_index=frame_index,
                _id=self._next_id,
                state=TrackletState.NEW if frame_index > self._initialization_threshold else TrackletState.ACTIVE
            )
            self._next_id += 1
            new_tracklets.append(new_tracklet)
            self._initiate(new_tracklet.id, detection)

            # Set initial tracklet feature if ReId model is used
            if objects_features is not None:
                new_tracklet.set(TrackletCommonData.APPEARANCE, objects_features[d_i])

        return new_tracklets

    def _update_tracklets(
        self,
        matched_tracklets: List[Tracklet],
        matched_detections: List[PredBBox],
        matched_object_features: Union[np.ndarray, List[Optional[np.ndarray]]],
        frame_index: int
    ) -> List[Tracklet]:
        """
        Updates tracklets with detections and object appearance features (if used).

        Args:
            matched_tracklets: List of matched tracklets
            matched_detections: List of matched detections
            frame_index: Current frame index

        Returns:
            List of updated tracklets
        """
        assert len(matched_tracklets) == len(matched_tracklets), \
            'Number of matched tracklets and number of matched detections do not match! ' \
            f'Got: {len(matched_tracklets)}, {len(matched_tracklets)}'

        for tracklet, det_bbox in zip(matched_tracklets, matched_detections):
            tracklet_bbox, _, _ = self._update(tracklet, det_bbox)
            new_bbox = det_bbox if self._use_observation_if_lost and tracklet.state != TrackletState.ACTIVE \
                else tracklet_bbox

            new_state = TrackletState.ACTIVE
            if tracklet.state == TrackletState.NEW and tracklet.total_matches + 1 < self._initialization_threshold:
                new_state = TrackletState.NEW
            tracklet.update(new_bbox, frame_index, state=new_state)

        self._update_appearances(
            tracklets=matched_tracklets,
            object_features=matched_object_features
        )

        return matched_tracklets

    def _update_appearances(self, tracklets: List[Tracklet], object_features: Union[np.ndarray, List[Optional[np.ndarray]]]) -> None:
        """
        Updates Tracklet appearance.

        Args:
            tracklets: Tracklets
            object_features: Detected objects features
        """
        if self._reid is None:
            return

        assert len(tracklets) == len(object_features), \
            'Number of matched tracklets and number of matched detection features do not match! ' \
            f'Got: {len(tracklets)}, {len(object_features)}'

        for i, tracklet in enumerate(tracklets):
            if object_features[i] is None:
                continue

            emb = tracklet.get(TrackletCommonData.APPEARANCE)
            if emb is None:
                emb = object_features[i]
            else:
                emb: np.ndarray
                emb = self._appearance_ema_momentum * emb + (1 - self._appearance_ema_momentum) * object_features[i]

            emb /= np.linalg.norm(emb)
            tracklet.set(TrackletCommonData.APPEARANCE, emb)

    def _handle_lost_tracklets(self, lost_tracklets: List[Tracklet]) -> None:
        """
        Handles lost tracklets:
        - Deletes tracklets that are lost for more than `self._remember_threshold` frames
        - Deletes new tracklets that are lost before initialized
        - Update lost tracklets position using the motion model

        Args:
            lost_tracklets: Lost tracklets that potentially should be deleted
        """
        for tracklet in lost_tracklets:
            if tracklet.lost_time > self._remember_threshold \
                    or tracklet.state == TrackletState.NEW:
                tracklet.state = TrackletState.DELETED
                self._delete(tracklet.id)
            else:
                tracklet_bbox, _, _ = self._missing(tracklet)
                tracklet.update(tracklet_bbox, tracklet.frame_index, state=TrackletState.LOST)

    def _perform_cmc(self, frame: np.ndarray, frame_index: int, bboxes: List[PredBBox]) -> List[PredBBox]:
        """
        Performs CMC on detected bounding boxes and updates motion filter states.

        Args:
            frame: Frame (image)
            frame_index: Current frame index
            bboxes: List of bounding boxes

        Returns:
            Updated bounding boxes
        """
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
        """
        Extracts features for each detected bounding box. Skips if ReID is not used

        Args:
            frame: Frame (image)
            frame_index: Current frame index
            bboxes: List of bounding boxes

        Returns:
            ReId features for each detected bounding box if ReId is used else None
        """
        if self._reid is None:
            # ReID is not used
            return None

        if self._reid_detection_threshold is not None:
            # Filter bboxes with high confidence
            # Remove bboxes with low confidence
            bboxes = [bbox for bbox in bboxes if bbox.conf >= self._reid_detection_threshold]

        return self._reid.extract_objects_features(frame, bboxes, frame_index=frame_index, scene=self._scene)

    def track(
        self,
        tracklets: List[Tracklet],
        detections: List[PredBBox],
        frame_index: int,
        frame: Optional[np.ndarray] = None
    ) -> List[Tracklet]:
        frame_index -= 1
        tracklets, prior_tracklet_bboxes = self._track_predict(tracklets, frame_index, frame=frame)
        objects_features = self._extract_reid_features(frame, frame_index, detections)
        return self._track(tracklets, prior_tracklet_bboxes, detections, frame_index,
                           objects_features=objects_features, frame=frame)

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
            frame: Pass frame in case CMC or appearance based association is used

        Returns:
            Tracks (active, lost, new, deleted, ...)
        """
