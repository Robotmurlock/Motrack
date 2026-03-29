"""
Utility functions for association algorithm.
"""
from motrack.tracker.tracklet import TrackletHistoryType


def filter_observations(history: TrackletHistoryType) -> TrackletHistoryType:
    """
    Remove all tracklet estimations. Estimations take place with
    last observation index if the observation is missing (missed detection).

    Disclaimer: This is coupled with MotionReId logic.

    Args:
        history:

    Returns:
        Tracklet observation history
    """
    last_frame_index: int = -1

    observation_history: TrackletHistoryType = []
    for tracklet_data in history:
        if last_frame_index == tracklet_data.frame_index:
            continue
        last_frame_index = tracklet_data.frame_index
        observation_history.append(tracklet_data)

    return observation_history

