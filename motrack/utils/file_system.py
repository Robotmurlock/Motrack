"""
File System custom support functions.
"""
import os
from typing import List, Optional
import re


def listdir(path: str, regex_filter: Optional[str] = None) -> List[str]:
    """
    Wrapper for `os.listdir` that ignore `.*` files (like `.DS_Store)

    Args:
        path: Directory path
        regex_filter: Filter files by regex

    Returns:
        Listed files (not hidden)
    """
    files = [p for p in os.listdir(path) if not p.startswith('.')]
    if regex_filter is not None:
        files = [p for p in os.listdir(path) if re.match(regex_filter, p)]
    return files
