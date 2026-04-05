"""
Interface for composing association algorithms with weights.

Notes:
    - Does not support cascaded algorithm;
    - Each algorithm needs to implement the `_form_cost_matrix`.
"""
from typing import List, Optional, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from motrack.library.cv import PredBBox
from motrack.tracker.matching.algorithms.base import AssociationAlgorithm
from motrack.tracker.matching.catalog import ASSOCIATION_CATALOG
from motrack.tracker.tracklet import Tracklet


class AssociationReferenceConfig(BaseModel):
    """
    Nested association reference used by compose configs.
    """

    model_config = ConfigDict(extra='forbid')

    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator('name')
    @classmethod
    def _normalize_name(cls, value: str) -> str:
        """
        Normalizes nested association names.

        Args:
            value: Raw association name.

        Returns:
            Normalized association name.
        """
        return value.lower()

    @field_validator('params', mode='before')
    @classmethod
    def _validate_params_dict(cls, value: Any) -> dict[str, Any]:
        """
        Validates nested params shape.

        Args:
            value: Raw nested params.

        Returns:
            Normalized params dictionary.
        """
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError(f'Expected association params to be a dictionary, but got {type(value).__name__}.')
        return value

    @model_validator(mode='after')
    def _validate_nested_association(self) -> 'AssociationReferenceConfig':
        """
        Validates nested association configs.

        Returns:
            Validated reference config.
        """
        try:
            config = ASSOCIATION_CATALOG.get_config(self.name).model_validate(self.params)
        except KeyError as exc:
            raise ValueError(f'Invalid association "{self.name}".') from exc
        self.params = config.model_dump()
        return self


@ASSOCIATION_CATALOG.register_config('compose')
class ComposeAssociationConfig(BaseModel):
    """
    Config for composed associations.
    """

    model_config = ConfigDict(extra='forbid')

    matchers: list[AssociationReferenceConfig] = Field(min_length=1)
    weights: list[float] = Field(min_length=1)
    fast_linear_assignment: bool = False

    @model_validator(mode='after')
    def _validate_matcher_weights(self) -> 'ComposeAssociationConfig':
        """
        Validates matcher and weight lengths.

        Returns:
            Validated compose config.
        """
        if len(self.matchers) != len(self.weights):
            raise ValueError(
                f'Expected the same number of compose matchers and weights, got '
                f'{len(self.matchers)} matchers and {len(self.weights)} weights.'
            )
        return self


@ASSOCIATION_CATALOG.register('compose')
class ComposeAssociationAlgorithm(AssociationAlgorithm):
    """
    Allows composition of multiple association matrix.
    """
    def __init__(self, config: ComposeAssociationConfig, matchers: List[AssociationAlgorithm]):
        """
        Args:
            matchers: List of association algorithms
            weights: Weight of each association algorithm cost matrix
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        super().__init__(fast_linear_assignment=config.fast_linear_assignment)

        weights = config.weights
        assert len(matchers) == len(weights), \
            f'Number of matchers and weights must be equal! Got {len(matchers)=} and {len(weights)=}'
        assert all(w >= 0 for w in weights), 'All weights must be non-negative!'
        assert any(w > 0 for w in weights), 'At least one weight must be positive!'

        self._matchers = matchers
        self._weights = weights

    def form_cost_matrix(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        object_features: Optional[np.ndarray] = None,
        tracklets: Optional[List[Tracklet]] = None
    ) -> np.ndarray:
        weighted_cost_matrix: Optional[np.ndarray] = None
        for matcher, weight in zip(self._matchers, self._weights):
            if weight == 0.0:
                continue

            cost_matrix = weight * matcher.form_cost_matrix(
                tracklet_estimations=tracklet_estimations,
                detections=detections,
                object_features=object_features,
                tracklets=tracklets
            )

            weighted_cost_matrix = cost_matrix if weighted_cost_matrix is None \
                else weighted_cost_matrix + cost_matrix

        return weighted_cost_matrix
