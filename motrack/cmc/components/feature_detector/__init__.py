"""
Feature detectors used to select points to establish correspondences between two frames.

A detector optionally produces descriptors alongside the point locations. Detectors that do
not (Shi-Tomasi) can only be used with tracking based correspondence (Lucas-Kanade), while
detectors that do (SIFT, ORB) can be used with either tracking or descriptor matching.
"""
from motrack.cmc.components.feature_detector.algorithms.base import DescriptorNorm, FeatureDetector
# noinspection PyUnresolvedReferences
from motrack.cmc.components.feature_detector.algorithms.orb import OrbFeatureDetector  # pylint: disable=unused-import
# noinspection PyUnresolvedReferences
from motrack.cmc.components.feature_detector.algorithms.shi_tomasi import ShiTomasiFeatureDetector  # pylint: disable=unused-import
# noinspection PyUnresolvedReferences
from motrack.cmc.components.feature_detector.algorithms.sift import SiftFeatureDetector  # pylint: disable=unused-import
from motrack.cmc.components.feature_detector.catalog import FEATURE_DETECTOR_CATALOG
from motrack.cmc.components.feature_detector.factory import feature_detector_factory
from motrack.cmc.components.feature_detector.utils import empty_points, pack_keypoints

FEATURE_DETECTOR_CATALOG.validate()
