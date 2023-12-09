"""
Wrapper of Bot-Sort KF.
"""
from typing import Tuple, Optional

import numpy as np

from motrack.filter.algorithms.base import State
from motrack.filter.algorithms.base import StateModelFilter
from motrack.filter.catalog import FILTER_CATALOG
from motrack.library.kalman_filter.botsort_kf import BotSortKalmanFilter


@FILTER_CATALOG.register('bot-sort')
class BotSortKalmanWrapFilter(StateModelFilter):
    """
    Wrapper for BotSortKalman filter for StateModelFilter interface.
    """
    def __init__(
        self,
        override_std_weight_position: Optional[float] = None
    ):
        self._kf = BotSortKalmanFilter(
            override_std_weight_position=override_std_weight_position
        )

    def initiate(self, measurement: np.ndarray) -> State:
        mean, covariance = self._kf.initiate(measurement)
        return mean, covariance

    def predict(self, state: State) -> State:
        mean, covariance = state
        mean_hat, covariance_hat = self._kf.predict(mean, covariance)
        return mean_hat, covariance_hat

    def multistep_predict(self, state: State, n_steps: int) -> State:
        mean, covariance = state
        mean_hat_list, cov_hat_list = [], []
        for _ in range(n_steps):
            mean, covariance = self.predict((mean, covariance))
            mean_hat_list.append(mean)
            cov_hat_list.append(covariance)

        mean_hat = np.stack(mean_hat_list)
        covariance_hat = np.stack(cov_hat_list)
        return mean_hat, covariance_hat

    def update(self, state: State, measurement: np.ndarray) -> State:
        mean_hat, covariance_hat = state
        mean, covariance = self._kf.update(mean_hat, covariance_hat, measurement)
        return mean, covariance

    def singlestep_to_multistep_state(self, state: State) -> State:
        mean, covariance = state
        return mean.unsqueeze(0), covariance.unsqueeze(0)

    def missing(self, state: State) -> State:
        return state  # Use prior instead of posterior

    def project(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        mean, covariance = state
        if len(mean.shape) == 1:
            # Single step projection
            mean_proj, covariance_proj = self._kf.project(mean, covariance)
        elif len(mean.shape) == 2:
            # Multistep (batch) projection
            batch_size = mean.shape[0]
            mean_proj_list, cov_proj_list = [], []
            for i in range(batch_size):
                mean_i, covariance_i = mean[i, :], covariance[i, :, :]
                mean_proj_i, covariance_proj_i = self._kf.project(mean_i, covariance_i)
                mean_proj_list.append(mean_proj_i)
                cov_proj_list.append(covariance_proj_i)

            mean_proj = np.stack(mean_proj_list)
            covariance_proj = np.stack(cov_proj_list)
        else:
            raise AssertionError(f'Invalid shape {mean.shape}')

        return mean_proj, np.diagonal(covariance_proj, axis1=-2, axis2=-1)

    def affine_transform(self, state: State, warp: np.ndarray) -> State:
        measurement, covariance = state
        L = np.kron(np.eye(4, dtype=np.float32), warp[:, :2])
        T = np.zeros(shape=8, dtype=np.float32)
        T[0:2] = warp[:, 2]
        measurement = L @ measurement + T
        covariance = L @ covariance @ L.T
        return measurement, covariance


def run_test() -> None:
    # TODO: Move to unit tests
    smf = BotSortKalmanWrapFilter()
    measurements = np.random.randn(10, 4)

    # Initiate test
    mean, cov = smf.initiate(measurements[0])
    assert mean.shape == (8,) and cov.shape == (8, 8)

    # Predict test
    mean_hat, cov_hat = smf.predict((mean, cov))
    assert mean_hat.shape == (8,) and cov_hat.shape == (8, 8)

    # Projected predict
    mean_hat_proj, cov_hat_proj = smf.project((mean_hat, cov_hat))
    assert mean_hat_proj.shape == (4,) and cov_hat_proj.shape == (4, 4)

    # Update test
    mean_updated, cov_updated = smf.update((mean, cov), measurements[1])
    assert mean_updated.shape == (8,) and cov_updated.shape == (8, 8)

    # Projected update
    mean_updated_proj, cov_updated_proj = smf.project((mean_updated, cov_updated))
    assert mean_updated_proj.shape == (4,) and cov_updated_proj.shape == (4, 4)

    # Multistep predict
    mean_multistep_hat, cov_multistep_hat = smf.multistep_predict((mean, cov), n_steps=5)
    assert mean_multistep_hat.shape == (5, 8) and cov_multistep_hat.shape == (5, 8, 8)

    # Projected multistep
    mean_multistep_hat_proj, cov_multistep_hat_proj = smf.project((mean_multistep_hat, cov_multistep_hat))
    assert mean_multistep_hat_proj.shape == (5, 4) and cov_multistep_hat_proj.shape == (5, 4, 4)


if __name__ == '__main__':
    run_test()
