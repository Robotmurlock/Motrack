"""
Motrack evaluation module.

Provides integrated MOT evaluation with HOTA, CLEAR, Identity, and Count
metrics. The metric implementations in this module are derived from the
TrackEval library and adapted to work within the Motrack pipeline.

Original source::

    @misc{luiten2020trackeval,
      author =       {Jonathon Luiten, Arne Hoffhues},
      title =        {TrackEval},
      howpublished = {\\url{https://github.com/JonathonLuiten/TrackEval}},
      year =         {2020}
    }
"""
import logging
import os
from typing import Any, Dict, List, Set, Tuple

from motrack.eval.io import load_mot_gt, load_mot_tracker
from motrack.eval.metrics import MetricBase
from motrack.eval.metrics.clear import CLEAR
from motrack.eval.metrics.count import Count
from motrack.eval.metrics.hota import HOTA
from motrack.eval.metrics.identity import Identity
from motrack.eval.preprocessing import preprocess_sequence
from motrack.eval.similarity import compute_box_ious

logger = logging.getLogger('Evaluation')

_METRIC_REGISTRY = {
    'HOTA': HOTA,
    'CLEAR': CLEAR,
    'Identity': Identity,
    'Count': Count,
}


def evaluate_tracker_output(
    gt_folder: str,
    tracker_folder: str,
    scene_names: List[str],
    seq_lengths: Dict[str, int],
    eval_classes: Set[int],
    distractor_classes: Set[int],
    metric_names: Tuple[str, ...] = ('HOTA', 'CLEAR', 'Identity', 'Count'),
) -> Dict[str, Any]:
    """
    Evaluates tracker output against ground truth.

    Args:
        gt_folder: Path to dataset split folder containing scene directories
                   with gt/gt.txt files.
        tracker_folder: Path to tracker output folder with {scene_name}.txt files.
        scene_names: List of scene names to evaluate.
        seq_lengths: Mapping from scene name to number of timesteps.
        eval_classes: Set of class IDs to evaluate.
        distractor_classes: Set of distractor class IDs.
        metric_names: Metric names to compute.

    Returns:
        Results dict with 'sequences' and 'combined' keys.
    """
    metrics = _instantiate_metrics(metric_names)

    per_sequence_results: Dict[str, Dict[str, Dict]] = {}

    for scene_name in scene_names:
        logger.info(f'Evaluating sequence: {scene_name}')
        num_timesteps = seq_lengths[scene_name]

        gt_path = os.path.join(gt_folder, scene_name, 'gt', 'gt.txt')
        tracker_path = os.path.join(tracker_folder, f'{scene_name}.txt')

        gt_data = load_mot_gt(gt_path, num_timesteps)
        tracker_data = load_mot_tracker(tracker_path, num_timesteps)

        # Merge and compute similarities
        raw_data = {**gt_data, **tracker_data}
        similarity_scores = []
        for t in range(num_timesteps):
            ious = compute_box_ious(raw_data['gt_dets'][t], raw_data['tracker_dets'][t])
            similarity_scores.append(ious)
        raw_data['similarity_scores'] = similarity_scores

        # Preprocess
        data = preprocess_sequence(raw_data, eval_classes, distractor_classes)

        # Compute metrics
        seq_results = {}
        for metric in metrics:
            seq_results[metric.name] = metric.eval_sequence(data)

        per_sequence_results[scene_name] = seq_results

    # Combine across sequences
    combined = {}
    for metric in metrics:
        all_seq_metric_res = {
            seq: per_sequence_results[seq][metric.name] for seq in scene_names
        }
        combined[metric.name] = metric.combine_sequences(all_seq_metric_res)

    return {
        'sequences': per_sequence_results,
        'combined': combined,
    }


def _instantiate_metrics(metric_names: Tuple[str, ...]) -> List[MetricBase]:
    metrics = []
    for name in metric_names:
        if name not in _METRIC_REGISTRY:
            available = sorted(_METRIC_REGISTRY.keys())
            raise ValueError(f'Unknown metric "{name}". Available: {available}')
        metrics.append(_METRIC_REGISTRY[name]())
    return metrics
