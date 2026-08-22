"""
Camera motion compensation interface.
"""
from motrack.cmc.factory import cmc_factory
from motrack.cmc.algorithms.base import CameraMotionCompensation, CMCContext
from motrack.cmc.algorithms.gmc_from_file import GMCFromFile
from motrack.cmc.algorithms.identity import IdentityCMC
