"""
Unit tests for CMC factory config validation.
"""
import unittest

from motrack.cmc.catalog import CMC_CATALOG
from motrack.cmc.factory import cmc_factory


class CmcFactoryTest(unittest.TestCase):
    """
    Tests for CMC factory validation.
    """

    def test_cmc_catalog_keys_match_config_keys(self) -> None:
        """
        Validates that CMC config models stay aligned with the CMC registry.
        """
        CMC_CATALOG.validate()
        self.assertEqual(set(CMC_CATALOG.config_keys), set(CMC_CATALOG.keys))

    def test_cmc_factory_validates_gmc_from_file_config(self) -> None:
        """
        Validates that gmc-from-file config is accepted with required params.
        """
        config = CMC_CATALOG.create_config('gmc-from-file', {'dirpath': '/tmp/gmc'}, params_label='cmc')
        self.assertIsNotNone(config)

    def test_cmc_factory_rejects_unknown_param(self) -> None:
        """
        Validates that unknown CMC params are rejected.
        """
        with self.assertRaisesRegex(ValueError, 'Invalid cmc'):
            CMC_CATALOG.create_config('gmc-from-file', {'dirpath': '/tmp', 'unknown': 1}, params_label='cmc')


if __name__ == '__main__':
    unittest.main()
