"""
Custom collection support.
"""
from collections import defaultdict
from typing import Callable, List, Any, Dict


def nesteddict() -> defaultdict:
    """
    Creates nested dict (like defaultdict(dict) but recursive)

    Returns:
        nested defaultdict(dict)
    """
    return defaultdict(nesteddict)


def defaultdict_to_dict(nested: defaultdict) -> dict:
    """
    Converts defaultdict to dict (recursively).

    Motivation: defaultdict is useful for construction but might give some side effects when performing queries.

    Args:
        nested: Dictionary to convert

    Returns:
        Converted dictionary
    """
    if not isinstance(nested, defaultdict):
        return nested

    return {k: defaultdict_to_dict(v) for k, v in nested.items()}


def group_by(items: List[Any], func: Callable) -> Dict[Any, List[Any]]:
    """
    Groups list items into buckets (list) by custom function.

    Args:
        items: List of items that need to be grouped into "buckets"
        func: Grouping function (can be seen as bucket hash)

    Returns:
        Mapping (bucket_hash -> bucket_of_items)
    """
    buckets = defaultdict(list)

    for item in items:
        item_hash = func(item)
        buckets[item_hash].append(item)

    return dict(buckets)


def unpack_n(items: list, n: int) -> List[list]:
    """
    Unpacks list of tuples with n items into n lists.
    Handles empty list edge case.

    Args:
        items: List of tuples
        n: Number of items per tuples

    Returns:
        List of unpacked lists.
    """
    if len(items) == 0:
        return n * [[]]

    return [list(x) for x in zip(*items)]

