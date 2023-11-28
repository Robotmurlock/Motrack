
from motrack.datasets.base import BaseDataset
from motrack.datasets.catalog import DATASET_CATALOG

# noinspection PyUnresolvedReferences
from motrack.datasets.mot import MOTDataset


def dataset_factory(name: str, params: dict) -> BaseDataset:
    """
    Dataset factory.

    Args:
        name: Dataset name (type)
        params: Dataset creation parameters

    Returns:
        Created dataset object
    """
    return DATASET_CATALOG[name](**params)
