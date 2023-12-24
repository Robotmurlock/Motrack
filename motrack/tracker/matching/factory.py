"""
Tracker association factory method.
Use `ASSOCIATION_CATALOG.register` to extend supported tracker association algorithms.
"""
# pylint: disable=unused-import
import copy
from typing import Dict, Any, List

from motrack.tracker.matching.algorithms.base import AssociationAlgorithm
# noinspection PyUnresolvedReferences
from motrack.tracker.matching.algorithms.biou import HungarianCBIoU, HungarianBIoU
from motrack.tracker.matching.algorithms.compose import ComposeAssociationAlgorithm
# noinspection PyUnresolvedReferences
from motrack.tracker.matching.algorithms.conf import HybridConfidenceAssociation
# noinspection PyUnresolvedReferences
from motrack.tracker.matching.algorithms.dcm import DCMIoU, MoveDCM
# noinspection PyUnresolvedReferences
from motrack.tracker.matching.algorithms.iou import IoUAssociation
# noinspection PyUnresolvedReferences
from motrack.tracker.matching.algorithms.move import Move
# noinspection PyUnresolvedReferences
from motrack.tracker.matching.algorithms.reid import ReIDIoUAssociation
from motrack.tracker.matching.catalog import ASSOCIATION_CATALOG


def association_factory(name: str, params: Dict[str, Any]) -> AssociationAlgorithm:
    """
    Association algorithm factory. Not case-sensitive.

    Allows weighted composition of multiple association algorithms
    (that implement the `_form_cost_matrix` method). Format:
    ```
    algorithm_name: compose
    algorithm_params:
        matchers:
            - name: iou
              params:
                match_threshold: 0.3
            - name: move
              params:
                match_threshold: 0.2
                motion_lambda: 0.1
        weights: [0.3, 0.7]
    ```

    Args:
        name: Algorithm name
        params: Parameters

    Returns:
        AssociationAlgorithm object
    """
    if name == 'compose':
        params = copy.deepcopy(params)

        assert 'matchers' in params, 'Expected "matchers" to be defined in parameters'
        matchers = params.pop('matchers')
        assert isinstance(matchers, list), f'Expected list for "matchers" by found {type(matchers)} instead!'

        assert 'weights' in params, 'Expected "weights" to be defined in parameters'
        weights = params.pop('weights')

        matcher_objs: List[AssociationAlgorithm] = []
        for matcher in matchers:
            combined_params = {**matcher['params'], **params}
            matcher_obj = association_factory(matcher['name'], combined_params)
            matcher_objs.append(matcher_obj)

        return ComposeAssociationAlgorithm(
            matchers=matcher_objs,
            weights=weights
        )

    return ASSOCIATION_CATALOG[name](**params)
