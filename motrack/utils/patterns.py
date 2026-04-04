"""
DynamicCatalog allows dynamic extensions of the factory methods.
"""
from typing import Any, Callable, List, Optional, Type

from pydantic import ValidationError


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


class DynamicConfigBasedCatalog(DynamicCatalog):
    """
    Dynamic catalog that keeps runtime classes and config models aligned.
    """

    def __init__(self):
        super().__init__()
        self._config_catalog = {}

    @property
    def config_keys(self) -> List[str]:
        """
        Returns:
            List of config catalog keys
        """
        return list(self._config_catalog.keys())

    def register_config(self, key: str) -> Callable[[Type], Type]:
        """
        Registers a config model to the config catalog.

        Args:
            key: Config key

        Returns:
            Config model
        """
        key = key.lower()

        if key in self._config_catalog:
            raise KeyError(f'Config key "{key}" already registered!')

        def config_register(config_cls: Type) -> Type:
            self._config_catalog[key] = config_cls
            return config_cls

        return config_register

    def get_config(self, key: str) -> Type:
        """
        Gets config model by key.

        Args:
            key: Config key

        Returns:
            Config model
        """
        key = key.lower()

        if key not in self._config_catalog:
            raise KeyError(f'Invalid config key "{key}". Registered: {self.config_keys}.')

        return self._config_catalog[key]

    def validate(self) -> None:
        """
        Validates runtime and config key alignment.

        Ignore Returns: This function returns None.

        Raises:
            RuntimeError: If runtime and config keys are not aligned.
        """
        runtime_keys = set(self.keys)
        config_keys = set(self.config_keys)
        if runtime_keys != config_keys:
            raise RuntimeError(
                f'Catalog keys {sorted(runtime_keys)} do not match config keys {sorted(config_keys)}.'
            )

    def create_config(
        self,
        key: str,
        params: Optional[dict],
        params_label: str,
        invalid_label: Optional[str] = None,
    ) -> Any:
        """
        Creates a validated config instance from plain factory params.

        Args:
            key: Registered algorithm key.
            params: Raw factory params.
            params_label: Label used in param type error messages.
            invalid_label: Label used in validation error messages.

        Returns:
            Validated config model instance.

        Raises:
            TypeError: If params are not a dictionary or None.
            RuntimeError: If runtime and config keys are not aligned.
            ValueError: If the key or params are invalid.
        """
        self.validate()

        normalized_key = key.lower()
        normalized_params = {} if params is None else params
        if not isinstance(normalized_params, dict):
            raise TypeError(
                f'Expected {params_label} params to be a dictionary, but got {type(normalized_params).__name__}.'
            )

        error_label = invalid_label or params_label

        try:
            config_cls = self.get_config(normalized_key)
        except KeyError as exc:
            raise ValueError(f'Invalid {error_label} "{normalized_key}".') from exc

        try:
            return config_cls.model_validate(normalized_params)
        except ValidationError as exc:
            raise ValueError(f'Invalid {error_label} "{normalized_key}": {exc}') from exc

    def create(
        self,
        key: str,
        params: Optional[dict],
        params_label: str,
        invalid_label: Optional[str] = None,
        config_transform: Optional[Callable[[Any], Any]] = None,
    ) -> Any:
        """
        Creates a runtime object from plain factory params.

        Args:
            key: Registered algorithm key.
            params: Raw factory params.
            params_label: Label used in param type error messages.
            invalid_label: Label used in validation error messages.
            config_transform: Optional post-processing hook for validated configs.

        Returns:
            Created runtime object.

        Raises:
            TypeError: If params are not a dictionary or None.
            RuntimeError: If runtime and config keys are not aligned.
            ValueError: If the key or params are invalid.
        """
        normalized_key = key.lower()
        error_label = invalid_label or params_label
        config = self.create_config(
            key=normalized_key,
            params=params,
            params_label=params_label,
            invalid_label=invalid_label,
        )

        if config_transform is not None:
            try:
                config = config_transform(config)
            except ValueError as exc:
                raise ValueError(f'Invalid {error_label} "{normalized_key}": {exc}') from exc

        runtime_cls = self[normalized_key]
        return runtime_cls(config)


def validate_catalog_factory_config(factory_config: Any, catalog: DynamicConfigBasedCatalog, label: str) -> Any:
    """
    Validates a nested factory config against a config-aware catalog.

    Args:
        factory_config: Nested factory config with `name` and `params` fields.
        catalog: Catalog that owns the target config model.
        label: Label used in validation error messages.

    Returns:
        Nested factory config with validated params.

    Raises:
        ValueError: If the nested key or params are invalid.
    """
    validated_config = catalog.create_config(factory_config.name, factory_config.params, params_label=label, invalid_label=label)
    return factory_config.model_copy(update={'params': validated_config.model_dump()})
