"""
Tracker. Supports:
- SORT
- ByteTrack
- SparseTrack
"""
from motrack.tracker.matching import AssociationAlgorithm, IoUAssociation
from motrack.tracker.trackers.algorithms.base import Tracker
from motrack.tracker.trackers.algorithms.byte import ByteTracker
from motrack.tracker.trackers.algorithms.sort import SortTracker
from motrack.tracker.trackers.algorithms.sparse import SparseTracker
from motrack.tracker.trackers.factory import tracker_factory
