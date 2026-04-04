"""
Tracker evaluation. Computes HOTA, CLEAR, Identity, and Count metrics.
"""
import logging
import os

import hydra
from motrack.common import conventions
from motrack.common.project import DANCETRACK_TRACKERS_CONFIG_PATH
from motrack.config_parser import GlobalConfig
from motrack.datasets import dataset_factory
from motrack.eval import evaluate_tracker_output
from motrack.eval.reporting import log_eval_results, dump_eval_results_json
from motrack.utils import pipeline
from omegaconf import DictConfig

logger = logging.getLogger('Tool-TrackerEvaluation')


@pipeline.task('eval')
def run_eval(cfg: GlobalConfig) -> None:
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

    dataset = dataset_factory(
        dataset_type=cfg.dataset.type,
        path=cfg.dataset.fullpath,
        params=cfg.dataset.params,
        test=False,
    )

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


@hydra.main(config_path=DANCETRACK_TRACKERS_CONFIG_PATH, config_name='movesort', version_base='1.1')
def main(cfg: DictConfig):
    # noinspection PyTypeChecker
    run_eval(cfg)


if __name__ == '__main__':
    main()
