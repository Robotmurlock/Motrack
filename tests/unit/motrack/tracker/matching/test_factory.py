"""
Unit tests for association factories.
"""
import unittest

from motrack.tracker.matching.factory import association_factory


ASSOCIATION_DEFAULT_INSTANTIABLE = [
    'iou',
    'adaptive-iou',
    'hmiou',
    'decay-iou',
    'biou',
    'cbiou',
    'reid',
    'long-term-reid',
    'hvc',
    'ocm',
    'robust-ocm',
    'ecm',
    'robust-ecm',
    'hybrid-conf',
    'distance',
    'dcm',
]


class AssociationFactoryTest(unittest.TestCase):
    """
    Tests for association factory validation.
    """

    def test_association_factory_creates_with_defaults(self) -> None:
        """
        Validates that all default-instantiable association algorithms
        can be created from empty params.
        """
        for name in ASSOCIATION_DEFAULT_INSTANTIABLE:
            with self.subTest(name=name):
                matcher = association_factory(name, {})
                self.assertIsNotNone(matcher)

    def test_association_factory_creates_compose(self) -> None:
        """
        Validates that compose can be created with explicit matchers and weights.
        """
        matcher = association_factory('compose', {
            'matchers': [
                {'name': 'iou', 'params': {}},
                {'name': 'hmiou', 'params': {}},
            ],
            'weights': [1.0, 0.5],
        })
        self.assertIsNotNone(matcher)

    @unittest.expectedFailure
    def test_association_factory_creates_move(self) -> None:
        """
        Validates that Move can be created with defaults.

        Known bug: Move passes empty matchers list to ComposeAssociationConfig
        which fails min_length=1 validation.
        """
        matcher = association_factory('move', {})
        self.assertIsNotNone(matcher)

    @unittest.expectedFailure
    def test_association_factory_creates_move_dcm(self) -> None:
        """
        Validates that Move-DCM can be created with defaults.

        Known bug: Transitively fails via Move.
        """
        matcher = association_factory('move-dcm', {})
        self.assertIsNotNone(matcher)

    @unittest.expectedFailure
    def test_association_factory_creates_reid_iou(self) -> None:
        """
        Validates that ReID-IoU can be created with defaults.

        Known bug: Same compose matchers=[] issue as Move.
        """
        matcher = association_factory('reid-iou', {})
        self.assertIsNotNone(matcher)

    def test_association_factory_rejects_unknown_param(self) -> None:
        """
        Validates that unknown association params are rejected.
        """
        with self.assertRaisesRegex(ValueError, 'Invalid association'):
            association_factory('iou', {'unknown': 1})

    def test_association_factory_rejects_invalid_compose_matchers(self) -> None:
        """
        Validates that compose matchers are recursively checked.
        """
        with self.assertRaisesRegex(ValueError, 'Invalid association "missing"'):
            association_factory(
                'compose',
                {
                    'matchers': [
                        {'name': 'missing', 'params': {}},
                    ],
                    'weights': [1.0],
                },
            )


if __name__ == '__main__':
    unittest.main()
