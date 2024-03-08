"""
Defines interface for motion model filter (predict - update)
"""
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional

import numpy as np

State = Any


class StateModelFilter(ABC):
    """
    Defines interface for motion model filter (predict - update)
    """
    @abstractmethod
    def initiate(self, measurement: np.ndarray, image: Optional[np.ndarray] = None) -> State:
        """
        Initializes model state with initial measurement.

        Args:
            measurement: Starting measurement
            image: Frame image

        Returns:
            Initial state
        """
        pass

    @abstractmethod
    def predict(self, state: State, image: Optional[np.ndarray] = None) -> State:
        """
        Predicts prior state.

        Args:
            state: Previous posterior state.
            image: Frame image

        Returns:
            Prior state (prediction)
        """
        pass

    @abstractmethod
    def multistep_predict(self, state: State, n_steps: int, image: Optional[np.ndarray] = None) -> State:
        """
        Estimates prior state for multiple steps ahead

        Args:
            state: Current state
            n_steps: Number of prediction steps
            image: Frame image

        Returns:
            Estimations for n_steps ahead
        """
        pass

    @abstractmethod
    def update(self, state: State, measurement: np.ndarray, image: Optional[np.ndarray] = None) -> State:
        """
        Updates state model based on new measurement.

        Args:
            state: Prior state.
            measurement: New measurement
            image: Frame image

        Returns:
            Posterior state.
        """
        pass

    @abstractmethod
    def missing(self, state: State, image: Optional[np.ndarray] = None) -> State:
        """
        Update state when measurement is missing (replacement for update function for that case).

        Args:
            state: State (prior)
            image: Frame image

        Returns:
            Updated state
        """

    @abstractmethod
    def project(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects state to the measurement space (estimation with uncertainty)

        Args:
            state: Model state

        Returns:
            Mean, Covariance
        """
        pass

    @abstractmethod
    def singlestep_to_multistep_state(self, state: State) -> State:
        """
        Converts singlestep state to multistep state. Used as an optimization.

        Args:
            state: Single step state

        Returns:
            Multistep state
        """
        pass

    def affine_transform(self, state: State, warp: np.ndarray) -> State:
        """
        Applies affine transform to the filter state.
        This is required only when used with CMC.

        Args:
            state: Current state
            warp: Affine transform matrix (warp)

        Returns:
            Warped state
        """
        raise NotImplementedError('This filter does not support affine transform!')
