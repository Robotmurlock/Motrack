"""
DynamicCatalog allows dynamic extensions of the factory methods.
"""
from typing import Callable, Type, List


class DynamicCatalog:
    """
    Dynamic catalog that is meant to be used with factory methods.
    Allows dynamic extension of the package factory method.
    """
    def __init__(self):
        self._catalog = {}

    @property
    def keys(self) -> List[str]:
        """
        Returns:
            List of catalog keys
        """
        return list(self._catalog.keys())

    def register(self, key: str) -> Callable[[Type], Type]:
        """
        Registers class to a catalog.

        Args:
            key: Class name

        Returns:
            Class
        """
        key = key.lower()

        if key in self._catalog:
            raise KeyError(f'Key "{key}" already registered!')

        def cls_register(cls: Type) -> Type:
            self._catalog[key] = cls
            return cls

        return cls_register

    def __getitem__(self, key: str) -> Type:
        key = key.lower()

        if key not in self._catalog:
            raise KeyError(f'Invalid key "{key}". Registered: {self.keys}.')

        return self._catalog[key]
