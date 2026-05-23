"""
Unit tests for filter factories.
"""
import unittest

from motrack.filter.factory import filter_factory


class FilterFactoryTest(unittest.TestCase):
    """
    Tests for filter factory validation.
    """

    def test_filter_factory_creates_with_defaults(self) -> None:
        """
        Validates that default-instantiable filters can be created.
        """
        cases = [
            ('bot-sort', {}),
            ('bot-sort', {'override_std_weight_position': 0.05}),
            ('no-motion', {}),
        ]
        for name, params in cases:
            with self.subTest(name=name, params=params):
                filter_object = filter_factory(name, params)
                self.assertIsNotNone(filter_object)

    def test_filter_factory_rejects_unknown_param(self) -> None:
        """
        Validates that unknown filter params are rejected.
        """
        with self.assertRaisesRegex(ValueError, 'Invalid filter'):
            filter_factory('bot-sort', {'unknown': 1})


if __name__ == '__main__':
    unittest.main()
