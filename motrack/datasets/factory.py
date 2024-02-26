"""
Dataset factory method.
Use `DATASET_CATALOG.register` to extend supported dataset formats.
"""
from motrack.datasets.base import BaseDataset
from motrack.datasets.catalog import DATASET_CATALOG
import copy

# noinspection PyUnresolvedReferences
from motrack.datasets.mot import MOTDataset  # pylint: disable=unused-import


def dataset_factory(dataset_type: str, path: str, params: dict, test: bool = False) -> BaseDataset:
    """
    Dataset factory.

    Args:
        dataset_type: Dataset name (type)
        path: Path is mandatory parameter
        params: Dataset creation parameters
        test: If True then ground truth annotations are not required

    Returns:
        Created dataset object
    """
    local_params = copy.deepcopy(params)  # Makes sure no inplace updates are performed on `params`

    if 'path' in local_params:
        raise ValueError('Key "path" must not be used in the params!')
    local_params['path'] = path
    if 'test' in local_params:
        raise ValueError('Key "test" must not be used in the params!')
    local_params['test'] = test

    return DATASET_CATALOG[dataset_type](**local_params)
