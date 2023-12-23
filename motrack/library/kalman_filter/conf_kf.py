"""
Implementation of ConfidenceFilter.
"""
import numpy as np
import scipy.linalg

# pylint: disable=pointless-string-statement
"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}


class ConfidenceKalmanFilter(object):
    """
    A simple Kalman filter for tracking detection confidence.

    The 2-dimensional state space

        c, vc
    """

    def __init__(
        self,
        initial_P_conf: float = 10.0,
        Q_conf: float = 1.0,
        Q_conf_velocity: float = 1e-3,
        R_conf: float = 100.0
    ):
        ndim, dt = 1, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Hyperparameters
        self._initial_P_conf = initial_P_conf
        self._Q_conf = Q_conf
        self._Q_conf_velocity = Q_conf_velocity
        self._R_conf = R_conf

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [self._initial_P_conf, 1000 * self._initial_P_conf]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [self._Q_conf]
        std_vel = [self._Q_conf_velocity]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        conf = mean[0]
        std = [self._R_conf * (1 - conf)]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)

        # noinspection PyUnresolvedReferences
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        # noinspection PyUnresolvedReferences
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


def run_test() -> None:
    measurements = np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.1, 0.1, 0.2, 0.8, 0.8, 0.9, 0.7, 0.6], dtype=np.float32).reshape(-1, 1)
    kf = ConfidenceKalmanFilter()
    mean, std = kf.initiate(measurements[0])

    for m in measurements[1:]:
        mean, std = kf.predict(mean, std)
        prior, _ = kf.project(mean, std)
        mean, std = kf.update(mean, std, m)
        posterior, _ = kf.project(mean, std)
        print(m, prior, posterior)


if __name__ == '__main__':
    run_test()
