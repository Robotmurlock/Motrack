"""
NoMotion filter (just uses the last detection).
"""
from typing import Tuple

import numpy as np

from motrack.filter.algorithms.base import StateModelFilter, State
from motrack.filter.catalog import FILTER_CATALOG
from motrack.library.numpy_utils.bbox import affine_transform


@FILTER_CATALOG.register('no-motion')
class NoMotionFilter(StateModelFilter):
    """
    Baseline filter trusts fully detector and is completely certain in detector accuracy.
    """
    def __init__(self, det_uncertainty = 1e-6):
        self._det_uncertainty = det_uncertainty

    def initiate(self, measurement: np.ndarray) -> State:
        return measurement

    def predict(self, state: State) -> State:
        measurement = state
        return measurement

    def multistep_predict(self, state: State, n_steps: int) -> State:
        measurement = state
        return np.stack([measurement for _ in range(n_steps)])

    def update(self, state: State, measurement: np.ndarray) -> State:
        _ = state  # ignore previous state
        return measurement

    def missing(self, state: State) -> State:
        return state

    def project(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        measurement = state
        det_mat = np.ones_like(measurement) * self._det_uncertainty
        return state, det_mat

    def singlestep_to_multistep_state(self, state: State) -> State:
        return state.unsqueeze(0)

    def affine_transform(self, state: State, warp: np.ndarray) -> State:
        measurement = state
        measurement[2:] = measurement[:2] + measurement[2:]  # xywh -> xyxy
        warped_measurement = affine_transform(measurement, warp)
        warped_measurement[2:] = warped_measurement[2:] - warped_measurement[:2]  # xyxy -> xywh
        return warped_measurement
