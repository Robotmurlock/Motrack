"""
High-level evaluation orchestration.

``run_eval`` is the library-level entry point used by both the CLI wrapper
and the optimizer. Dataset construction is pluggable via ``dataset_builder``;
the default reproduces today's behavior.
"""
import logging
import os
from typing import Any, Dict

from motrack.common import conventions
from motrack.config_parser import GlobalConfig
from motrack.eval import evaluate_tracker_output
from motrack.eval.reporting import log_eval_results, dump_eval_results_json
from motrack.tools.dataset_builder import DatasetBuilder, default_dataset_builder

logger = logging.getLogger('Tool-TrackerEvaluation')


def run_eval(
    cfg: GlobalConfig,
    dataset_builder: DatasetBuilder = default_dataset_builder,
) -> Dict[str, Any]:
    """Evaluate tracker output against ground truth and dump results.

    Args:
        cfg: Global config. Must point to an existing experiment directory
            with tracker output already produced.
        dataset_builder: Pluggable dataset construction.

    Returns:
        Results dict with ``sequences`` and ``combined`` keys.
    """
    assert cfg.inference.split != 'test', \
        'Cannot evaluate on test split — ground-truth is typically unavailable.'
    assert os.path.exists(cfg.experiment_path), \
        f'Experiment path "{cfg.experiment_path}" does not exist. Run inference first.'

    tracker_output = conventions.get_tracker_output_path(
        cfg.experiment_path,
        cfg.eval.eval_output,
    )
    assert os.path.exists(tracker_output), \
        f'Tracker output path "{tracker_output}" does not exist.'

    dataset = dataset_builder(cfg)

    seq_lengths = {
        scene: dataset.get_scene_info(scene).seqlength
        for scene in dataset.scenes
    }

    logger.info(f'Evaluating tracker output at "{tracker_output}".')

    results = evaluate_tracker_output(
        gt_folder=cfg.dataset.fullpath,
        tracker_folder=tracker_output,
        scene_names=dataset.scenes,
        seq_lengths=seq_lengths,
        eval_classes=set(cfg.eval.eval_classes),
        distractor_classes=set(cfg.eval.distractor_classes),
    )

    log_eval_results(results, dataset.scenes)

    json_path = conventions.get_eval_results_path(cfg.experiment_path)
    dump_eval_results_json(results, json_path)

    return results
