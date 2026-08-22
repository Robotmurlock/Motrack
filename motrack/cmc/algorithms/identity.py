"""
Identity (no-op) camera motion compensation.
"""
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict

from motrack.cmc.algorithms.base import CameraMotionCompensation, CMCContext
from motrack.cmc.catalog import CMC_CATALOG
from motrack.cmc.components.warp import identity_warp


@CMC_CATALOG.register_config('identity')
class IdentityCMCConfig(BaseModel):
    """
    Config for the identity CMC. Takes no parameters.
    """

    model_config = ConfigDict(extra='forbid')


@CMC_CATALOG.register('identity')
class IdentityCMC(CameraMotionCompensation):
    """
    Always returns an identity warp.

    This is a control rather than a useful algorithm: running a tracker with `identity`
    must produce exactly the same output as running it without CMC at all, which makes it
    a cheap regression test for the CMC plumbing. It is also useful for isolating the
    interface overhead in runtime measurements.
    """
    requires_image: ClassVar[bool] = False

    def __init__(self, config: IdentityCMCConfig):
        """
        Args:
            config: Empty config
        """
        _ = config

    def apply(self, ctx: CMCContext) -> np.ndarray:
        _ = ctx
        return identity_warp()
