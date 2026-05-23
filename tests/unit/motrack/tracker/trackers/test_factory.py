"""
Unit tests for tracker factories.
"""
import unittest

from motrack.tracker.trackers.algorithms.byte import ByteTracker
from motrack.tracker.trackers.algorithms.sort import SortTracker
from motrack.tracker.trackers.algorithms.sparse import SparseTracker
from motrack.tracker.trackers.catalog import TRACKER_CATALOG
from motrack.tracker.trackers.factory import tracker_factory


MINIMAL_FILTER = {'name': 'bot-sort', 'params': {}}


class TrackerFactoryTest(unittest.TestCase):
    """
    Tests for tracker factory validation.
    """

    def test_tracker_factory_catalog_keys_match_config_keys(self) -> None:
        """
        Validates that tracker config models stay aligned with the tracker registry.
        """
        TRACKER_CATALOG.validate()
        self.assertEqual(set(TRACKER_CATALOG.config_keys), set(TRACKER_CATALOG.keys))

    def test_tracker_factory_creates_sort(self) -> None:
        """
        Validates SORT creation with minimal params.
        """
        tracker = tracker_factory('sort', {'filter': MINIMAL_FILTER})
        self.assertIsInstance(tracker, SortTracker)

    def test_tracker_factory_creates_sort_with_full_params(self) -> None:
        """
        Validates SORT creation with explicit nested component params.
        """
        tracker = tracker_factory(
            'sort',
            {
                'filter': {
                    'name': 'bot-sort',
                    'params': {'override_std_weight_position': 0.05},
                },
                'matcher': {
                    'name': 'iou',
                    'params': {'match_threshold': 0.25},
                },
                'initialization_threshold': 3,
                'remember_threshold': 30,
            },
        )
        self.assertIsInstance(tracker, SortTracker)

    def test_tracker_factory_creates_byte(self) -> None:
        """
        Validates ByteTrack creation with minimal params.
        """
        tracker = tracker_factory('byte', {'filter': MINIMAL_FILTER})
        self.assertIsInstance(tracker, ByteTracker)

    def test_tracker_factory_creates_byte_with_detection_threshold(self) -> None:
        """
        Validates that ByteTrack detection_threshold propagates
        to new_tracklet_detection_threshold when not set explicitly.
        """
        tracker = tracker_factory(
            'byte',
            {
                'filter': MINIMAL_FILTER,
                'detection_threshold': 0.7,
            },
        )
        self.assertIsInstance(tracker, ByteTracker)
        self.assertEqual(0.7, tracker._new_tracklet_detection_threshold)

    def test_tracker_factory_creates_sparse(self) -> None:
        """
        Validates SparseTrack creation with minimal params.
        """
        tracker = tracker_factory('sparse', {'filter': MINIMAL_FILTER})
        self.assertIsInstance(tracker, SparseTracker)

    def test_tracker_factory_rejects_invalid_nested_component_name(self) -> None:
        """
        Validates that nested component names are checked before tracker creation.
        """
        with self.assertRaisesRegex(ValueError, 'Invalid tracker config'):
            tracker_factory(
                'sort',
                {
                    'filter': {
                        'name': 'missing',
                        'params': {},
                    },
                },
            )

    def test_tracker_factory_rejects_invalid_thresholds(self) -> None:
        """
        Validates that numeric constraints are enforced by the config model.
        """
        with self.assertRaisesRegex(ValueError, 'Invalid tracker config'):
            tracker_factory(
                'byte',
                {
                    'filter': MINIMAL_FILTER,
                    'detection_threshold': 1.5,
                },
            )


if __name__ == '__main__':
    unittest.main()
