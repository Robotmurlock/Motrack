"""
Logging Utility
"""
import numpy as np
import torch

import logging

logger = logging.getLogger('UtilsLogging')


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configures logger

    Args:
        level: Logging level
    """
    np.set_printoptions(formatter={'float': '{:0.3f}'.format})
    torch.set_printoptions(sci_mode=False, precision=3)
    logging.basicConfig(
        level=logging.getLevelName(level),
        format='%(asctime)s [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
