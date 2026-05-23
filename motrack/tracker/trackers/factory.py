"""
Tracker factory method.
Use `TRACKER_CATALOG.register` to extend supported tracker algorithms.
"""
from typing import Any

from motrack.tracker.trackers.algorithms.base import Tracker
# noinspection PyUnresolvedReferences
from motrack.tracker.trackers.algorithms.byte import ByteTracker  # pylint: disable=unused-import
# noinspection PyUnresolvedReferences
from motrack.tracker.trackers.algorithms.sort import SortTracker  # pylint: disable=unused-import
# noinspection PyUnresolvedReferences
from motrack.tracker.trackers.algorithms.sparse import SparseTracker  # pylint: disable=unused-import
from motrack.tracker.trackers.catalog import TRACKER_CATALOG
from motrack.utils.patterns import validate_catalog_factory_config


def _validate_nested_tracker_components(tracker_config: Any) -> Any:
    """
    Validates nested tracker component configs after top-level model creation.

    Args:
        tracker_config: Tracker config instance created by the catalog-registered model.

    Returns:
        Tracker config with validated nested component params.

    Raises:
        ValueError: If a nested component name or params are invalid.
    """
    from motrack.cmc.catalog import CMC_CATALOG
    from motrack.filter.catalog import FILTER_CATALOG
    from motrack.reid.catalog import REID_CATALOG
    from motrack.tracker.matching.catalog import ASSOCIATION_CATALOG

    nested_catalogs = {
        'filter': FILTER_CATALOG,
        'cmc': CMC_CATALOG,
        'reid': REID_CATALOG,
        'matcher': ASSOCIATION_CATALOG,
        'high_matcher': ASSOCIATION_CATALOG,
        'low_matcher': ASSOCIATION_CATALOG,
        'new_matcher': ASSOCIATION_CATALOG,
    }

    for field_name, catalog in nested_catalogs.items():
        factory_config = getattr(tracker_config, field_name, None)
        if factory_config is None:
            continue

        validated_config = validate_catalog_factory_config(factory_config, catalog, field_name)
        setattr(tracker_config, field_name, validated_config)

    if (
        hasattr(tracker_config, 'detection_threshold')
        and getattr(tracker_config, 'new_tracklet_detection_threshold', None) is None
    ):
        tracker_config.new_tracklet_detection_threshold = tracker_config.detection_threshold

    return tracker_config


def tracker_factory(name: str, params: dict) -> Tracker:
    """
    Tracker factory

    Args:
        name: tracker name
        params: tracker params

    Returns:
        Tracker object

    Raises:
        TypeError: If tracker params are not a dictionary or None.
        RuntimeError: If tracker config models and registered trackers are out of sync.
        ValueError: If tracker params are invalid.
    """
    return TRACKER_CATALOG.create(
        name,
        params,
        params_label='tracker',
        invalid_label='tracker config for',
        config_transform=_validate_nested_tracker_components,
    )
