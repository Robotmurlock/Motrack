"""
Interface for composing association algorithms with weights.

Notes:
    - Does not support cascaded algorithm;
    - Each algorithm needs to implement the `_form_cost_matrix`.
"""
from typing import List, Optional

import numpy as np

from motrack.library.cv import PredBBox
from motrack.tracker.matching.algorithms.base import AssociationAlgorithm
from motrack.tracker.matching.catalog import ASSOCIATION_CATALOG
from motrack.tracker.tracklet import Tracklet


@ASSOCIATION_CATALOG.register('compose')
class ComposeAssociationAlgorithm(AssociationAlgorithm):
    """
    Allows composition of multiple association matrix.
    """
    def __init__(
        self,
        matchers: List[AssociationAlgorithm],
        weights: List[float],
        fast_linear_assignment: bool = False,
    ):
        """
        Args:
            matchers: List of association algorithms
            weights: Weight of each association algorithm cost matrix
            fast_linear_assignment: Use greedy algorithm for linear assignment
                - This might be more efficient in case of large cost matrix
        """
        super().__init__(fast_linear_assignment=fast_linear_assignment)

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
