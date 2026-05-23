"""
MOT-format file loading for evaluation.

Reads ground-truth and tracker output files in the standard MOT Challenge
text format into per-timestep numpy arrays suitable for metric computation.
All coordinates are kept in raw pixel units (not normalized).
"""
import csv
import os
from typing import Dict, List, Tuple

import numpy as np


def load_mot_gt(path: str, num_timesteps: int) -> Dict[str, List[np.ndarray]]:
    """
    Loads a ground-truth file in MOT Challenge format.

    GT columns: frame_id, id, bb_left, bb_top, bb_width, bb_height,
                conf/zero_marked, class_id, visibility

    Args:
        path: Path to the GT text file.
        num_timesteps: Total number of frames in the sequence.

    Returns:
        Dict with per-timestep arrays:
        - gt_ids: List[np.ndarray] of shape (num_dets,) int
        - gt_dets: List[np.ndarray] of shape (num_dets, 4) float32 xywh
        - gt_classes: List[np.ndarray] of shape (num_dets,) int
        - gt_extras: List[Dict[str, np.ndarray]] with 'zero_marked' key
    """
    raw = _load_text_file(path)

    gt_ids = [None] * num_timesteps
    gt_dets = [None] * num_timesteps
    gt_classes = [None] * num_timesteps
    gt_extras = [None] * num_timesteps

    for t in range(num_timesteps):
        time_key = str(t + 1)
        if time_key in raw:
            time_data = np.asarray(raw[time_key], dtype=np.float32)
            gt_ids[t] = np.atleast_1d(time_data[:, 1]).astype(int)
            gt_dets[t] = np.atleast_2d(time_data[:, 2:6])
            gt_classes[t] = np.atleast_1d(time_data[:, 7]).astype(int) if time_data.shape[1] >= 8 else np.ones(len(gt_ids[t]), dtype=int)
            gt_extras[t] = {'zero_marked': np.atleast_1d(time_data[:, 6]).astype(int)}
        else:
            gt_ids[t] = np.empty(0, dtype=int)
            gt_dets[t] = np.empty((0, 4), dtype=np.float32)
            gt_classes[t] = np.empty(0, dtype=int)
            gt_extras[t] = {'zero_marked': np.empty(0, dtype=int)}

    return {
        'gt_ids': gt_ids,
        'gt_dets': gt_dets,
        'gt_classes': gt_classes,
        'gt_extras': gt_extras,
        'num_timesteps': num_timesteps,
    }


def load_mot_tracker(path: str, num_timesteps: int) -> Dict[str, List[np.ndarray]]:
    """
    Loads a tracker output file in MOT Challenge format.

    Tracker columns: frame_id, id, bb_left, bb_top, bb_width, bb_height,
                     conf, -1, -1, -1

    Args:
        path: Path to the tracker text file.
        num_timesteps: Total number of frames in the sequence.

    Returns:
        Dict with per-timestep arrays:
        - tracker_ids: List[np.ndarray] of shape (num_dets,) int
        - tracker_dets: List[np.ndarray] of shape (num_dets, 4) float32 xywh
        - tracker_confidences: List[np.ndarray] of shape (num_dets,) float32
    """
    raw = _load_text_file(path)

    tracker_ids = [None] * num_timesteps
    tracker_dets = [None] * num_timesteps
    tracker_confidences = [None] * num_timesteps

    for t in range(num_timesteps):
        time_key = str(t + 1)
        if time_key in raw:
            time_data = np.asarray(raw[time_key], dtype=np.float32)
            tracker_ids[t] = np.atleast_1d(time_data[:, 1]).astype(int)
            tracker_dets[t] = np.atleast_2d(time_data[:, 2:6])
            tracker_confidences[t] = np.atleast_1d(time_data[:, 6])
        else:
            tracker_ids[t] = np.empty(0, dtype=int)
            tracker_dets[t] = np.empty((0, 4), dtype=np.float32)
            tracker_confidences[t] = np.empty(0, dtype=np.float32)

    return {
        'tracker_ids': tracker_ids,
        'tracker_dets': tracker_dets,
        'tracker_confidences': tracker_confidences,
        'num_timesteps': num_timesteps,
    }


def _load_text_file(path: str) -> Dict[str, List[List[str]]]:
    """
    Loads a MOT-format text file, grouping rows by timestep.

    Returns:
        Dict mapping timestep string (1-indexed) to list of row values.
    """
    data: Dict[str, List[List[str]]] = {}
    with open(path, 'r', encoding='utf-8') as fp:
        fp.seek(0, os.SEEK_END)
        if not fp.tell():
            return data
        fp.seek(0)
        dialect = csv.Sniffer().sniff(fp.readline(), delimiters=',')
        dialect.skipinitialspace = True
        fp.seek(0)
        for row in csv.reader(fp, dialect):
            if row[-1] in '':
                row = row[:-1]
            timestep = str(int(float(row[0])))
            if timestep in data:
                data[timestep].append(row)
            else:
                data[timestep] = [row]
    return data
