"""
Backfill ``scenes_evaluated`` and ``wall_time_s`` on TrialResult records that
predate the C2 budget-logging change. Optionally re-logs the same metrics to
the corresponding MLflow run (matched by ``config_hash`` tag).

Reconstruction sources:
- ``scenes_evaluated``: count of ``sequences`` keys in ``eval_results.json``
  for the trial's ``config_hash``. Always equals the full-eval scene count
  for Optuna trials (each Optuna trial = one full eval).
- ``wall_time_s``: sum of per-scene ``e2e_total_s`` from ``fps_stats.json``.
  This captures inference + tracking time. Optuna sampler overhead is
  excluded (negligible vs. a 25-scene full eval).

Usage:
    python tools/migrate/backfill_pre_c2_budget.py \\
        --studies sort-tpe sort-tpe-prior sort-tpe-multivariate \\
                  sort-tpe-gamma030 sort-tpe-neic64 sort-tpe-neic64-gamma010 \\
                  sort-random \\
        [--mlflow]                # also re-log to MLflow

Requires the project's optimization_results.json files to be on disk under
``$MASTER_PATH/dancetrack/<study>/val/optimizations/<study_name>/``.
"""
import argparse
import json
import os
import sys
from typing import Optional, Tuple

DATASET_ROOT = '/media/home/motrack-outputs/dancetrack'


def _scenes_evaluated(infer_dir: str, config_hash: str) -> int:
    eval_path = os.path.join(infer_dir, config_hash, 'eval_results.json')
    if not os.path.exists(eval_path):
        return 0
    with open(eval_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return len(data.get('sequences', {}))


def _wall_time_s(infer_dir: str, config_hash: str) -> float:
    fps_path = os.path.join(infer_dir, config_hash, 'fps_stats.json')
    if not os.path.exists(fps_path):
        return 0.0
    with open(fps_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    scenes = data.get('scenes') or []
    return float(sum(s.get('e2e_total_s', 0.0) for s in scenes))


def backfill_study(study_dir: str) -> Tuple[int, int, float]:
    """Backfill all optimization_results.json under ``study_dir``.

    Returns ``(trials_updated, total_scenes, total_wall_s)``.
    """
    val_dir = os.path.join(study_dir, 'val')
    infer_dir = os.path.join(val_dir, 'inference')
    opt_results = []
    for root, _, files in os.walk(os.path.join(val_dir, 'optimizations')):
        for fn in files:
            if fn == 'optimization_results.json':
                opt_results.append(os.path.join(root, fn))

    total_trials = 0
    total_scenes = 0
    total_wall = 0.0
    for path in opt_results:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        modified = False
        for t in data['all_trials']:
            ch = t.get('config_hash')
            if not ch:
                continue
            n_scenes = _scenes_evaluated(infer_dir, ch)
            wall = _wall_time_s(infer_dir, ch)
            if n_scenes > 0 and t.get('scenes_evaluated', 0) == 0:
                t['scenes_evaluated'] = n_scenes
                modified = True
            if wall > 0 and t.get('wall_time_s', 0.0) == 0.0:
                t['wall_time_s'] = wall
                modified = True
            total_scenes += n_scenes
            total_wall += wall
            total_trials += 1
        # Mirror onto best_trial
        best = data.get('best_trial')
        if best is not None and 'config_hash' in best:
            ch = best['config_hash']
            best['scenes_evaluated'] = _scenes_evaluated(infer_dir, ch)
            best['wall_time_s'] = _wall_time_s(infer_dir, ch)
        if modified:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f'  updated: {path}')
    return total_trials, total_scenes, total_wall


_KEY_METRICS = {
    ('HOTA', 'HOTA'): 'HOTA', ('HOTA', 'DetA'): 'DetA', ('HOTA', 'AssA'): 'AssA',
    ('HOTA', 'DetRe'): 'DetRe', ('HOTA', 'DetPr'): 'DetPr',
    ('HOTA', 'AssRe'): 'AssRe', ('HOTA', 'AssPr'): 'AssPr', ('HOTA', 'LocA'): 'LocA',
    ('CLEAR', 'MOTA'): 'MOTA', ('CLEAR', 'MOTP'): 'MOTP', ('CLEAR', 'IDSW'): 'IDSW',
    ('Identity', 'IDF1'): 'IDF1', ('Identity', 'IDR'): 'IDR', ('Identity', 'IDP'): 'IDP',
}


def _eval_metrics(eval_path: str) -> dict:
    """Extract scalar combined metrics from eval_results.json."""
    if not os.path.exists(eval_path):
        return {}
    with open(eval_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out = {}
    combined = data.get('combined', {})
    for (group, field), name in _KEY_METRICS.items():
        v = combined.get(group, {}).get(field)
        if v is None:
            continue
        if isinstance(v, list):
            v = v[0] if v else None
        if isinstance(v, (int, float)):
            out[name] = float(v)
    return out


def _flatten_yaml(d, prefix='', out=None):
    if out is None:
        out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            _flatten_yaml(v, f'{prefix}.{k}' if prefix else k, out)
    else:
        out[prefix] = str(d) if d is not None else 'None'
    return out


def relog_to_mlflow(study_dir: str, tracking_uri: str) -> int:
    """Create MLflow experiment + runs for a historical study from disk.

    For each trial: creates one MLflow run named after its config_hash,
    populated with eval metrics, fps stats, params (algorithm.params.*),
    and the reconstructed budget metrics (scenes_evaluated, trial_wall_time_s).
    """
    import yaml
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    val_dir = os.path.join(study_dir, 'val')
    infer_dir = os.path.join(val_dir, 'inference')
    study_name = os.path.basename(study_dir)
    experiment_name = f'dancetrack/{study_name}/val'
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        eid = client.create_experiment(experiment_name)
        print(f'  created MLflow experiment id={eid}: {experiment_name}')
    else:
        eid = experiment.experiment_id

    opt_results = []
    for root, _, files in os.walk(os.path.join(val_dir, 'optimizations')):
        for fn in files:
            if fn == 'optimization_results.json':
                opt_results.append(os.path.join(root, fn))

    n_runs_created = 0
    for path in opt_results:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        study_optuna_name = data.get('study_name', study_name)
        for t in data['all_trials']:
            ch = t.get('config_hash')
            if not ch:
                continue
            existing = client.search_runs(
                experiment_ids=[eid],
                filter_string=f"tags.config_hash = '{ch}'",
                max_results=1,
            )
            if existing:
                continue  # already logged
            trial_dir = os.path.join(infer_dir, ch)
            cfg_path = os.path.join(trial_dir, 'config.yaml')
            eval_path = os.path.join(trial_dir, 'eval_results.json')
            fps_path = os.path.join(trial_dir, 'fps_stats.json')

            metrics = _eval_metrics(eval_path)
            scenes = _scenes_evaluated(infer_dir, ch)
            wall = _wall_time_s(infer_dir, ch)
            if scenes > 0:
                metrics['scenes_evaluated'] = float(scenes)
            if wall > 0:
                metrics['trial_wall_time_s'] = float(wall)
            if os.path.exists(fps_path):
                with open(fps_path, 'r', encoding='utf-8') as f:
                    fps = json.load(f)
                if 'e2e_fps' in fps:
                    metrics['e2e_fps'] = float(fps['e2e_fps'])
                if 'association_fps' in fps:
                    metrics['association_fps'] = float(fps['association_fps'])

            params = {}
            if os.path.exists(cfg_path):
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                params = _flatten_yaml(cfg.get('algorithm', {}).get('params', {}), 'algorithm.params')
                params['algorithm.name'] = cfg.get('algorithm', {}).get('name', 'unknown')

            tags = {
                'config_hash': ch,
                'dataset.type': 'dancetrack',
                'experiment': study_name,
                'split': 'val',
                'optuna.study_name': study_optuna_name,
                'optuna.trial_number': str(t.get('number', -1)),
                'backfilled': 'true',
            }
            run = client.create_run(
                experiment_id=eid,
                run_name=f'{ch} (trial-{t.get("number", -1)})',
                tags=tags,
            )
            run_id = run.info.run_id
            for k, v in params.items():
                # MLflow param values are capped at 6000 chars; strings only.
                client.log_param(run_id, k[:250], str(v)[:6000])
            for k, v in metrics.items():
                client.log_metric(run_id, k, v)
            for art_path in (cfg_path, eval_path, fps_path):
                if os.path.exists(art_path):
                    try:
                        client.log_artifact(run_id, art_path)
                    except Exception:
                        pass
            client.set_terminated(run_id, status='FINISHED')
            n_runs_created += 1
    return n_runs_created


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--studies', nargs='+', required=True,
                    help='Study experiment dirnames under dancetrack/, e.g. sort-tpe sort-random')
    ap.add_argument('--mlflow', action='store_true',
                    help='Also re-log metrics to the matching MLflow runs')
    ap.add_argument('--mlflow-uri', default='http://motrack-mlflow:5000')
    args = ap.parse_args()

    grand_trials = 0
    grand_scenes = 0
    grand_wall = 0.0
    for study in args.studies:
        d = os.path.join(DATASET_ROOT, study)
        if not os.path.isdir(d):
            print(f'[skip] not found: {d}')
            continue
        print(f'\n=== {study} ===')
        n_t, n_s, n_w = backfill_study(d)
        print(f'  trials touched: {n_t}, total scenes: {n_s}, total wall: {n_w/60:.1f} min')
        grand_trials += n_t
        grand_scenes += n_s
        grand_wall += n_w
        if args.mlflow:
            try:
                n_logged = relog_to_mlflow(d, args.mlflow_uri)
                print(f'  MLflow metrics pushed for {n_logged} runs')
            except Exception as e:
                print(f'  MLflow push failed: {e}')

    print(f'\n=== TOTAL ===')
    print(f'  trials: {grand_trials}')
    print(f'  scenes: {grand_scenes}')
    print(f'  wall:   {grand_wall/60:.1f} min ({grand_wall/3600:.1f} h)')


if __name__ == '__main__':
    sys.exit(main() or 0)
