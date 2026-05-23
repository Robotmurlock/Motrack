"""
Unit tests for ReID factory config validation.
"""
import unittest

from motrack.reid.catalog import REID_CATALOG


class ReIDFactoryTest(unittest.TestCase):
    """
    Tests for ReID factory validation.
    """

    def test_reid_catalog_keys_match_config_keys(self) -> None:
        """
        Validates that ReID config models stay aligned with the ReID registry.
        """
        REID_CATALOG.validate()
        self.assertEqual(set(REID_CATALOG.config_keys), set(REID_CATALOG.keys))

    def test_reid_factory_validates_fastreid_onnx_config(self) -> None:
        """
        Validates that fastreid-onnx config is accepted with required params.
        """
        config = REID_CATALOG.create_config(
            'fastreid-onnx',
            {'model_path': '/tmp/model.onnx'},
            params_label='reid',
        )
        self.assertIsNotNone(config)

    def test_reid_factory_rejects_unknown_param(self) -> None:
        """
        Validates that unknown ReID params are rejected.
        """
        with self.assertRaisesRegex(ValueError, 'Invalid reid'):
            REID_CATALOG.create_config(
                'fastreid-onnx',
                {'model_path': '/tmp/model.onnx', 'unknown': 1},
                params_label='reid',
            )


if __name__ == '__main__':
    unittest.main()
