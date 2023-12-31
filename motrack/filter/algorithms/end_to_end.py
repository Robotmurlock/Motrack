"""
Wrapper for Motrack-motion end-to-end filter.
"""
from typing import Tuple

import numpy as np
import torch
from motrack.filter.algorithms.base import StateModelFilter, State
from motrack.filter.catalog import FILTER_CATALOG
from motrack_motion.datasets import transforms
from motrack_motion.filter import BufferedE2EFilter
from motrack_motion.models import model_factory
from torch import nn


@FILTER_CATALOG.register('motrack-motion-end-to-end')
class MotrackMotionFilterEndToEnd(StateModelFilter):
    """
    Motrack-motion framework end-to-end filters
    """
    def __init__(
        self,
        model_type: str,
        model_params: dict,
        checkpoint_path: str,
        transform_type: str,
        transform_params: dict,
        accelerator: str,

        buffer_size: int,
        buffer_min_size: int = 1,
        buffer_min_history: int = 5,

        recursive_inverse: bool = False
    ):
        """
        Args:
            model_type: Model type (architecture name)
            model_params: Model parameters
            checkpoint_path: Checkpoint path (PLT checkpoint)
            transform_type: Transform type
            transform_params: Transform parameters
            accelerator: Accelerator (cpu/gpu)
            buffer_size: Maximum buffer size
            buffer_min_size: Minimum buffer size in order to have a meaningful model input
            buffer_min_history: Minimum number of items in buffer before deletinf old items
            recursive_inverse: Performs recursive inverse for transform function (cumsum)
        """
        transform = transforms.transform_factory(transform_type, transform_params)

        self._core = BufferedE2EFilter(
            model=self._load_model_from_pl_checkpoint(model_type, model_params, checkpoint_path),
            transform=transform,
            accelerator=accelerator,

            buffer_size=buffer_size,
            buffer_min_size=buffer_min_size,
            buffer_min_history=buffer_min_history,
            dtype=torch.float32,

            recursive_inverse=recursive_inverse
        )

    def _load_model_from_pl_checkpoint(
        self,
        model_type: str,
        model_params: dict,
        checkpoint_path: str
    ) -> nn.Module:
        """
        Loads model weights from a PLT checkpoint.

        Args:
            model_type: Model type (architecture name)
            model_params: Model parameters
            checkpoint_path: Checkpoint path (PLT checkpoint)

        Returns:
            Loaded model
        """
        model = model_factory(model_type, model_params)
        state_dict = torch.load(checkpoint_path)
        model_state_dict = {k.replace('_model.', '', 1): v for k, v in state_dict['state_dict'].items()}
        model.load_state_dict(model_state_dict)
        return model

    def initiate(self, measurement: np.ndarray) -> State:
        measurement = torch.from_numpy(measurement)
        return self._core.initiate(measurement)

    def predict(self, state: State) -> State:
        return self._core.predict(state)

    def multistep_predict(self, state: State, n_steps: int) -> State:
        return self._core.multistep_predict(state, n_steps=n_steps)

    def update(self, state: State, measurement: np.ndarray) -> State:
        measurement = torch.from_numpy(measurement)
        return self._core.update(state, measurement)

    def missing(self, state: State) -> State:
        return self._core.missing(state)

    def project(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        mean, cov = self._core.project(state)
        mean, cov = mean.cpu().numpy(), cov.cpu().numpy()
        return mean, cov

    def singlestep_to_multistep_state(self, state: State) -> State:
        return self.singlestep_to_multistep_state(state)
