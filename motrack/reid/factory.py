"""
ReID factory method.
Use `REID_CATALOG.register` to extend supported ReID algorithms.
"""
# noinspection PyUnresolvedReferences
from motrack.reid.algorithms.fastreid_onnx import FastReIDOnnx  # pylint: disable=unused-import
from motrack.reid.algorithms.base import BaseReID
from motrack.reid.catalog import REID_CATALOG

REID_CATALOG.validate()


def reid_inference_factory(
    name: str,
    params: dict,
) -> BaseReID:
    """
    Creates ReID inference.

    Args:
        name: ReID inference name (type)
        params: CTor parameters

    Returns:
        Initialized ReID inference

    Raises:
        TypeError: If ReID params are not a dictionary or None.
        ValueError: If ReID params are invalid.
    """
    return REID_CATALOG.create(name, params, params_label='reid')
