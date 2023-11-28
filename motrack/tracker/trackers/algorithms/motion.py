"""
Implementation of Filter Sort tracker
"""
from typing import Tuple, Optional

import numpy as np
from abc import ABC
import torch

from nodetracker.filter import filter_factory
from nodetracker.library.cv.bbox import PredBBox, BBox
from nodetracker.tracker.trackers.base import Tracker
from nodetracker.tracker.tracklet import Tracklet


class MotionBasedTracker(Tracker, ABC):
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
    ):
        """
        Args:
            filter_name: Filter name
            filter_params: Filter params
        """


        self._filter = filter_factory(filter_name, filter_params)

        # State
        self._filter_states = {}

    @staticmethod
    def _raw_to_bbox(tracklet: Tracklet, raw: torch.Tensor, conf: Optional[float] = None) -> PredBBox:
        """
        Converts raw tensor to PredBBox for tracked object.

        Args:
            tracklet: Tracklet
            raw: raw bbox coords
            conf: BBox confidence

        Returns:
            PredBBox
        """
        bbox_raw = raw.numpy().tolist()
        return PredBBox.create(
            bbox=BBox.from_yxwh(*bbox_raw, clip=False),
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
        measurement = torch.from_numpy(detection.as_numpy_yxwh(dtype=np.float32))
        state = self._filter.initiate(measurement)
        self._filter_states[tracklet_id] = state

    def _predict(self, tracklet: Tracklet) -> Tuple[PredBBox, torch.Tensor, torch.Tensor]:
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

    def _update(self, tracklet: Tracklet, detection: PredBBox) -> Tuple[PredBBox, torch.Tensor, torch.Tensor]:
        """
        Estimates object posterior position based on the matched detection.

        Args:
            tracklet: Tracklet (required to fetch state, and bbox info)

        Returns:
            Motion model posterior estimation
        """
        measurement = torch.from_numpy(detection.as_numpy_yxwh())

        state = self._filter_states[tracklet.id]
        state = self._filter.update(state, measurement)
        posterior_mean, posterior_std = self._filter.project(state)
        self._filter_states[tracklet.id] = state
        bbox = self._raw_to_bbox(tracklet, posterior_mean, conf=detection.conf)
        return bbox, posterior_mean, posterior_std

    def _missing(self, tracklet: Tracklet) -> Tuple[PredBBox, torch.Tensor, torch.Tensor]:
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
