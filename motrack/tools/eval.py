"""
High-level evaluation orchestration.

``run_eval`` is the library-level entry point used by both the CLI wrapper
and the optimizer. Dataset construction is pluggable via ``dataset_builder``;
the default reproduces today's behavior.
"""
import logging
import os
import re
from typing import Any, Dict, List, Optional

from motrack.common import conventions
from motrack.config_parser import GlobalConfig
from motrack.eval import evaluate_tracker_output
from motrack.eval.reporting import log_eval_results, dump_eval_results_json
from motrack.tools.dataset_builder import DatasetBuilder, default_dataset_builder

logger = logging.getLogger('Tool-TrackerEvaluation')


def run_eval(
    cfg: GlobalConfig,
    dataset_builder: DatasetBuilder = default_dataset_builder,
    scenes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate tracker output against ground truth and dump results.

    Args:
        cfg: Global config. Must point to an existing experiment directory
            with tracker output already produced.
        dataset_builder: Pluggable dataset construction.
        scenes: Optional explicit list of scenes to evaluate. When ``None``
            (default), the scenes selected by ``dataset_filter.scene_pattern``
            are evaluated. When provided, only those scenes are scored — useful
            for low-fidelity evaluation in multi-fidelity optimizers. Scenes
            must be a subset of ``dataset.scenes``.

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

    if scenes is None:
        # Inference and postprocessing both filter on this pattern, so evaluation has to as
        # well: otherwise a filtered run is scored against scenes it was never asked to track.
        pattern = cfg.dataset_filter.scene_pattern
        eval_scenes = [scene for scene in dataset.scenes if re.match(pattern, scene)]
        assert eval_scenes, f'Scene pattern "{pattern}" matched none of the {len(dataset.scenes)} dataset scenes.'
    else:
        unknown = set(scenes) - set(dataset.scenes)
        assert not unknown, f'Unknown scenes requested: {sorted(unknown)}'
        eval_scenes = list(scenes)

    seq_lengths = {
        scene: dataset.get_scene_info(scene).seqlength
        for scene in eval_scenes
    }

    logger.info(f'Evaluating tracker output at "{tracker_output}" ({len(eval_scenes)} scenes).')

    results = evaluate_tracker_output(
        gt_folder=cfg.dataset.fullpath,
        tracker_folder=tracker_output,
        scene_names=eval_scenes,
        seq_lengths=seq_lengths,
        eval_classes=set(cfg.eval.eval_classes),
        distractor_classes=set(cfg.eval.distractor_classes),
    )

    log_eval_results(results, eval_scenes)

    json_path = conventions.get_eval_results_path(cfg.experiment_path)
    dump_eval_results_json(results, json_path)

    return results
