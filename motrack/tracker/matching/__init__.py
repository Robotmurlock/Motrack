"""
Tracker association algorithms interface.
"""
from motrack.tracker.matching.algorithms.base import AssociationAlgorithm
from motrack.tracker.matching.algorithms.iou import HungarianAlgorithmIOU
from motrack.tracker.matching.algorithms.dcm import DCMIoU, MoveDCM
from motrack.tracker.matching.algorithms.move import Move
from motrack.tracker.matching.factory import association_factory
