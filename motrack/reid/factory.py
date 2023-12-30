"""
ReID factory method.
Use `REID_CATALOG.register` to extend supported ReID algorithms.
"""
from motrack.reid.algorithms.base import BaseReID
# noinspection PyUnresolvedReferences
from motrack.reid.algorithms.fastreid_onnx import FastReIDOnnx  # pylint: disable=unused-import
from motrack.reid.catalog import REID_CATALOG


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
    """
    return REID_CATALOG[name](**params)
