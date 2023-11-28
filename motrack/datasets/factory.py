
from motrack.datasets.base import BaseDataset
from motrack.datasets.catalog import DATASET_CATALOG

# noinspection PyUnresolvedReferences
from motrack.datasets.mot import MOTDataset


def dataset_factory(name: str, path: str, params: dict) -> BaseDataset:
    """
    Dataset factory.

    Args:
        name: Dataset name (type)
        path: Path is mandatory parameter
        params: Dataset creation parameters

    Returns:
        Created dataset object
    """
    if 'path' in params:
        raise ValueError('Key "path" must not be used in the params!')
    params['path'] = path
    return DATASET_CATALOG[name](**params)
