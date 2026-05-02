"""
MFGCS sampler-params dataclass.

Validated by :func:`pipeline_factory` when ``cfg.optimizer.sampler ==
'mfgcs'``. The content arrives via ``cfg.optimizer.sampler_params`` as a
plain nested dict (Hydra leaves ``Dict[str, Any]`` unconverted), so
nested ``{type, params}`` blocks for ``scene_sampler`` /
``coordinate_optimizer`` and the ``shrink`` sub-config are coerced to
their typed wrappers in :meth:`__post_init__`.
"""
from dataclasses import dataclass, field

from motrack.config_parser import FactorySpec


@dataclass
class MFGCSShrinkConfig:
    """Search-space shrinking settings for MFGCS.

    After accepting a coordinate move, the parameter's window is narrowed
    around the accepted value to focus subsequent sweeps.
    """
    enabled: bool = True
    radius_frac: float = 0.25      # numeric: ±r·(B−A); log floats: same in log space
    window_size: int = 3           # ordered-discrete: ±k indices around accepted index


@dataclass
class MFGCSParams:
    """Multi-Fidelity Greedy Coordinate Search settings.

    Pluggable components use the project's standard ``{type, params}``
    factory shape so variant-specific knobs stay self-contained.
    """
    # Defaults intentionally carry no ``params`` keys — variant-specific
    # defaults live in the variant param dataclass; injecting them here would
    # leak mismatched keys (e.g. ``grid`` into the random optimizer) when the
    # YAML overrides ``type`` without overriding ``params``.
    scene_sampler: FactorySpec = field(
        default_factory=lambda: FactorySpec(type='random', params={})
    )
    coordinate_optimizer: FactorySpec = field(
        default_factory=lambda: FactorySpec(type='grid', params={})
    )
    max_sweeps: int = 10
    # Hard cap on full-fidelity trials (bootstrap counts as 1). Provides a
    # complementary stopping criterion to ``max_sweeps`` when the search space
    # is large — sweeping every coord at ``max_sweeps`` would otherwise blow
    # the budget. Set to ``0`` to disable.
    max_trials: int = 100
    bootstrap_full_eval: bool = True
    # Terminate as soon as a full sweep produces no accepted move. Default-on:
    # for greedy coordinate search, a barren sweep means the windows weren't
    # shrunk and ``current`` is unchanged, so subsequent sweeps would mostly
    # rerun the same exploration with only the scene subset varying. Set to
    # ``False`` to force exactly ``max_sweeps`` sweeps for ablation purposes.
    early_stop: bool = True
    # Minimum HOTA improvement on the full-fidelity gate to count as an accept.
    # Set just above numerical-noise floor so genuine small accepts (~1e-3 to
    # 1e-4 range observed on SORT/DanceTrack) are kept. ``1e-2`` is too strict
    # — empirically rejects every accept and the algorithm makes no progress.
    accept_threshold: float = 1e-4
    # Drop a parameter from the active search space after this many consecutive
    # sweeps with no accepted move on it. ``0`` disables the dropout. Reduces
    # wasted budget on parameters that have converged or are insensitive.
    drop_after_barren_sweeps: int = 2
    shrink: MFGCSShrinkConfig = field(default_factory=MFGCSShrinkConfig)

    def __post_init__(self) -> None:
        if isinstance(self.scene_sampler, dict):
            self.scene_sampler = FactorySpec(**self.scene_sampler)
        if isinstance(self.coordinate_optimizer, dict):
            self.coordinate_optimizer = FactorySpec(**self.coordinate_optimizer)
        if isinstance(self.shrink, dict):
            self.shrink = MFGCSShrinkConfig(**self.shrink)
