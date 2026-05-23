"""
Generate a Markdown comparison report from all Optuna optimization runs
found under the configured dataset and split.

Uses the same Hydra config as ``tools.optimize``, so paths and MLflow
settings are inherited automatically.

Usage:
    python -m tools.analysis.report_optimization \
        --config-name optimization/report_per_family
"""
import glob
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Postgres VARCHAR(250) caps MLflow run_name; keep some headroom.
MAX_RUN_NAME_LEN = 200

import hydra
import matplotlib.pyplot as plt
import numpy as np
import yaml

from motrack.common import conventions
from motrack.common.project import DANCETRACK_TRACKERS_CONFIG_PATH
from motrack.config_parser import GlobalConfig
from motrack.tools.mlflow_logger import mlflow, is_mlflow_enabled
from motrack.utils import pipeline
from motrack.eval.results import EvalResults
from motrack.tools.optimization import OptimizationResults, TrialResult

logger = logging.getLogger('ReportOptimization')

REPORT_FILENAME = 'opt_report.md'
HOTA_CHART_BASENAME = 'opt_report_hota'
METRICS_CHART_BASENAME = 'opt_report_metrics'
FPS_CHART_BASENAME = 'opt_report_hota_vs_fps'
BUDGET_CHART_BASENAME = 'opt_report_hota_vs_budget'
WALLTIME_CHART_BASENAME = 'opt_report_hota_vs_walltime'

# Efficiency threshold: first trial within 1% of the best HOTA
EFFICIENCY_RATIO = 0.99

# Default TPE startup trials (random sampling before TPE kicks in)
DEFAULT_N_STARTUP_TRIALS = 10

# Group name used when no explicit groups are configured (single-section report).
DEFAULT_GROUP_NAME = 'all'


@dataclass
class TrialMetrics:
    """Per-trial metrics loaded from eval_results and fps_stats."""
    hota: float
    idf1: Optional[float] = None
    mota: Optional[float] = None
    idsw: Optional[int] = None
    association_fps: Optional[float] = None


@dataclass
class StudyData:
    """Optimization results paired with per-trial metrics."""
    results: OptimizationResults
    split_path: str
    trial_metrics: Dict[int, TrialMetrics] = field(default_factory=dict)
    n_startup_trials: Optional[int] = None

    @property
    def display_name(self) -> str:
        # split_path = {master}/{dataset}/{experiment}/{split}
        experiment = os.path.basename(os.path.dirname(self.split_path))
        return experiment

    @property
    def completed_trials(self) -> List[TrialResult]:
        return [t for t in self.results.all_trials if t.state == 'COMPLETE']


def discover_optimization_results(
    master_path: str,
    dataset_type: str,
    split: str,
    dataset_name: Optional[str] = None,
) -> List[str]:
    """Find all optimization_results.json, excluding test experiments."""
    ds_name = conventions.get_dataset_name(dataset_type, dataset_name)
    pattern = os.path.join(
        master_path, ds_name, '*', split,
        conventions.OPTIMIZATIONS_DIRNAME, '*',
        conventions.OPTIMIZATION_RESULTS_FILENAME,
    )
    paths = sorted(glob.glob(pattern))
    paths = [p for p in paths if 'test' not in os.path.basename(os.path.dirname(p)).lower()]
    return paths


def _load_trial_metrics(run_dir: str, hota: float) -> TrialMetrics:
    """Load metrics for a single trial from its run directory."""
    metrics = TrialMetrics(hota=hota)

    eval_path = conventions.get_eval_results_path(run_dir)
    if os.path.exists(eval_path):
        er = EvalResults.load(eval_path)
        idf1 = er.combined.get('Identity', {}).get('IDF1')
        if idf1 is not None:
            metrics.idf1 = float(idf1)
        mota = er.combined.get('CLEAR', {}).get('MOTA')
        if mota is not None:
            metrics.mota = float(mota)
        idsw = er.combined.get('CLEAR', {}).get('IDSW')
        if idsw is not None:
            metrics.idsw = int(idsw)

    fps_path = conventions.get_fps_stats_path(run_dir)
    if os.path.exists(fps_path):
        with open(fps_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        if 'association_fps' in stats:
            metrics.association_fps = float(stats['association_fps'])

    return metrics


def _extract_sampler_params(opt: dict) -> dict:
    """Read ``optimizer.sampler_params`` with a fallback for legacy snapshots.

    Studies optimized before the sampler-instantiation refactor stored
    MFGCS knobs under a separate ``optimizer.mfgcs`` block instead of
    ``optimizer.sampler_params``. When loading those snapshots, fold the
    legacy block in so downstream readers see a single shape.
    """
    sampler_params = dict(opt.get('sampler_params') or {})
    if opt.get('sampler') == 'mfgcs' and not sampler_params:
        legacy = opt.get('mfgcs')
        if legacy:
            return dict(legacy)
    return sampler_params


def _load_n_startup_trials(split_path: str, results: OptimizationResults) -> Optional[int]:
    """Read n_startup_trials from the first trial's config snapshot."""
    completed = [t for t in results.all_trials if t.state == 'COMPLETE']
    if not completed:
        return None

    config_path = conventions.get_config_snapshot_path(
        os.path.join(conventions.get_inference_path(split_path), completed[0].config_hash),
    )
    if not os.path.exists(config_path):
        return None

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    opt = cfg.get('optimizer') or {}
    sampler = opt.get('sampler', '')
    if sampler not in ('tpe', 'warm_tpe'):
        return None

    return _extract_sampler_params(opt).get('n_startup_trials', DEFAULT_N_STARTUP_TRIALS)


def load_study_data(result_path: str) -> StudyData:
    """Load optimization results and per-trial metrics."""
    results = OptimizationResults.load(result_path)
    split_path = result_path.rsplit(f'/{conventions.OPTIMIZATIONS_DIRNAME}/', 1)[0]

    trial_metrics: Dict[int, TrialMetrics] = {}
    for trial in results.all_trials:
        if trial.state != 'COMPLETE':
            continue
        run_dir = os.path.join(conventions.get_inference_path(split_path), trial.config_hash)
        trial_metrics[trial.number] = _load_trial_metrics(run_dir, trial.value)

    n_startup = _load_n_startup_trials(split_path, results)

    return StudyData(
        results=results,
        split_path=split_path,
        trial_metrics=trial_metrics,
        n_startup_trials=n_startup,
    )


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def running_best(trials: List[TrialResult]) -> np.ndarray:
    """Cumulative max HOTA across trials (ordered by trial number)."""
    sorted_trials = sorted(trials, key=lambda t: t.number)
    values = np.array([t.value for t in sorted_trials])
    return np.maximum.accumulate(values)


def first_trial_above(trials: List[TrialResult], threshold: float) -> Optional[int]:
    """First trial number where running best HOTA >= threshold."""
    sorted_trials = sorted(trials, key=lambda t: t.number)
    best_so_far = -np.inf
    for t in sorted_trials:
        best_so_far = max(best_so_far, t.value)
        if best_so_far >= threshold:
            return t.number
    return None


def _cumulative(trials: List[TrialResult], attr: str) -> List[float]:
    """Return the cumulative sum of ``trial.<attr>`` ordered by trial number."""
    sorted_trials = sorted(trials, key=lambda t: t.number)
    out: List[float] = []
    total = 0.0
    for t in sorted_trials:
        total += float(getattr(t, attr, 0) or 0.0)
        out.append(total)
    return out


def total_wall_time(trials: List[TrialResult]) -> float:
    """Total optimization wall-time in seconds, summed across trials."""
    return float(sum(float(getattr(t, 'wall_time_s', 0.0) or 0.0) for t in trials))


def first_budget_above(
    trials: List[TrialResult],
    threshold: float,
    attr: str,
) -> Optional[float]:
    """Cumulative budget (scenes or wall-time) at which running-best HOTA reaches ``threshold``."""
    sorted_trials = sorted(trials, key=lambda t: t.number)
    best_so_far = -np.inf
    cumulative = 0.0
    for t in sorted_trials:
        cumulative += float(getattr(t, attr, 0) or 0.0)
        best_so_far = max(best_so_far, t.value)
        if best_so_far >= threshold:
            return cumulative
    return None


def _format_walltime(seconds: float) -> str:
    """Render seconds as a compact human-readable string."""
    if seconds <= 0:
        return 'N/A'
    if seconds < 60:
        return f'{seconds:.0f}s'
    if seconds < 3600:
        return f'{seconds / 60.0:.1f}m'
    return f'{seconds / 3600.0:.2f}h'


def _group_studies(
    studies: List[StudyData],
    groups: Dict[str, List[str]],
) -> List[Tuple[str, List[StudyData]]]:
    """Bucket studies into named groups using their display names.

    Empty ``groups`` maps to the single default group containing every study,
    preserving the script's pre-grouping behaviour. Studies named in a group
    that don't match any discovered study are warned about; matched studies
    that don't appear in any group are reported once (info-level)."""
    if not groups:
        return [(DEFAULT_GROUP_NAME, studies)]

    by_name: Dict[str, StudyData] = {s.display_name: s for s in studies}
    out: List[Tuple[str, List[StudyData]]] = []
    matched: set = set()
    for group_name, names in groups.items():
        bucket: List[StudyData] = []
        for n in names:
            s = by_name.get(n)
            if s is None:
                logger.warning(f'Group "{group_name}": no study with display_name "{n}"')
                continue
            bucket.append(s)
            matched.add(n)
        if bucket:
            out.append((group_name, bucket))
        else:
            logger.warning(f'Group "{group_name}" matched zero studies; skipping.')

    leftover = sorted(set(by_name) - matched)
    if leftover:
        logger.info(f'Studies not in any group (skipped): {leftover}')
    return out


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def build_summary_table(studies: List[StudyData]) -> str:
    header = (
        "| Study | Trials | Best HOTA | IDF1 | MOTA | IDSW "
        "| Best Trial # | Trial @ 99% | Assoc FPS | Wall-time |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    rows = []
    for s in studies:
        trials = s.completed_trials
        n_trials = len(trials)
        best = s.results.best_trial
        threshold = best.value * EFFICIENCY_RATIO
        trial_eff = first_trial_above(trials, threshold)
        trial_eff_str = str(trial_eff) if trial_eff is not None else "N/A"

        m = s.trial_metrics.get(best.number)
        idf1_str = f"{m.idf1:.4f}" if m and m.idf1 is not None else "N/A"
        mota_str = f"{m.mota:.4f}" if m and m.mota is not None else "N/A"
        idsw_str = str(m.idsw) if m and m.idsw is not None else "N/A"
        fps_str = f"{m.association_fps:.1f}" if m and m.association_fps is not None else "N/A"
        wall_str = _format_walltime(total_wall_time(trials))

        rows.append(
            f"| {s.display_name} | {n_trials} | {best.value:.4f} "
            f"| {idf1_str} | {mota_str} | {idsw_str} "
            f"| {best.number} | {trial_eff_str} | {fps_str} | {wall_str} |"
        )
    return "\n".join([header, sep] + rows)


def build_best_params_section(studies: List[StudyData]) -> str:
    sections = []
    for s in studies:
        best = s.results.best_trial
        lines = [f"### {s.display_name}", ""]
        lines.append(f"**HOTA: {best.value:.4f}** (trial #{best.number})")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|---|---|")
        for k, v in sorted(best.params.items()):
            if isinstance(v, float):
                lines.append(f"| `{k}` | {v:.4f} |")
            else:
                lines.append(f"| `{k}` | {v} |")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _draw_startup_lines(ax: plt.Axes, studies: List[StudyData]) -> None:
    """Draw vertical lines marking the end of TPE startup (random) trials."""
    drawn: set = set()
    for s in studies:
        n = s.n_startup_trials
        if n is None or n in drawn:
            continue
        drawn.add(n)
        ax.axvline(
            x=n, color='grey', linestyle='--', linewidth=1, alpha=0.6,
            label=f'TPE startup ({n})',
        )


def plot_trials(studies: List[StudyData], output_path: str) -> None:
    """Per-trial HOTA scatter with running best overlay."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for s in studies:
        trials = s.completed_trials
        sorted_trials = sorted(trials, key=lambda t: t.number)
        xs = [t.number for t in sorted_trials]
        ys = [t.value for t in sorted_trials]
        color = ax.scatter(xs, ys, s=18, alpha=0.35, label=s.display_name).get_facecolors()[0]
        cum_best = running_best(trials)
        ax.plot(xs, cum_best, linewidth=2, color=color, alpha=0.9)

    _draw_startup_lines(ax, studies)

    ax.set_xlabel("Trial")
    ax.set_ylabel("HOTA")
    ax.set_title("Per-Trial HOTA (dots) with Running Best (line)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_trial_metrics(studies: List[StudyData], output_path: str) -> None:
    """2x2 chart: trial vs HOTA/IDF1/MOTA/IDSW with running best line."""
    metrics = [
        ('hota', 'HOTA', False),
        ('idf1', 'IDF1', False),
        ('mota', 'MOTA', False),
        ('idsw', 'IDSW', True),  # minimize
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (attr, label, minimize) in zip(axes.flat, metrics):
        for s in studies:
            trials = s.completed_trials
            sorted_trials = sorted(trials, key=lambda t: t.number)
            xs, ys = [], []
            for t in sorted_trials:
                m = s.trial_metrics.get(t.number)
                if m is None:
                    continue
                val = getattr(m, attr, None)
                if val is not None:
                    xs.append(t.number)
                    ys.append(val)
            if not xs:
                continue

            color = ax.scatter(xs, ys, s=18, alpha=0.35, label=s.display_name).get_facecolors()[0]
            values = np.array(ys)
            if minimize:
                cum_best = np.minimum.accumulate(values)
            else:
                cum_best = np.maximum.accumulate(values)
            ax.plot(xs, cum_best, linewidth=2, color=color, alpha=0.9)

        _draw_startup_lines(ax, studies)

        ax.set_xlabel("Trial")
        ax.set_ylabel(label)
        ax.set_title(f"Trial vs {label}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_hota_vs_budget(studies: List[StudyData], output_path: str) -> None:
    """Running-best HOTA vs cumulative scenes evaluated.

    Skipped (no file produced) when no study reports nonzero scene budgets;
    that is the case for legacy results captured before scene-budget logging.
    """
    has_budget = any(
        any(getattr(t, 'scenes_evaluated', 0) for t in s.completed_trials)
        for s in studies
    )
    if not has_budget:
        logger.info('No scene-budget data available. Skipping HOTA vs budget chart.')
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for s in studies:
        trials = s.completed_trials
        if not trials:
            continue
        xs = _cumulative(trials, 'scenes_evaluated')
        ys = running_best(trials)
        ax.plot(xs, ys, marker='o', linewidth=2, alpha=0.85, label=s.display_name)

    ax.set_xlabel('Cumulative scenes evaluated')
    ax.set_ylabel('Best HOTA so far')
    ax.set_title('HOTA vs Scene-Evaluation Budget')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_hota_vs_walltime(studies: List[StudyData], output_path: str) -> None:
    """Running-best HOTA vs cumulative optimization wall-time."""
    has_walltime = any(
        any(getattr(t, 'wall_time_s', 0.0) for t in s.completed_trials)
        for s in studies
    )
    if not has_walltime:
        logger.info('No wall-time data available. Skipping HOTA vs wall-time chart.')
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for s in studies:
        trials = s.completed_trials
        if not trials:
            continue
        xs = _cumulative(trials, 'wall_time_s')
        ys = running_best(trials)
        ax.plot(xs, ys, marker='o', linewidth=2, alpha=0.85, label=s.display_name)

    ax.set_xlabel('Cumulative wall-time (s)')
    ax.set_ylabel('Best HOTA so far')
    ax.set_title('HOTA vs Wall-Time Budget')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_hota_vs_fps(studies: List[StudyData], output_path: str) -> None:
    """HOTA vs association FPS, one point per study (the best-HOTA trial).

    Per-trial association FPS is sensitive to transient machine load
    during the run (e.g. background processes pulling CPU), which can
    introduce per-trial outliers that are not properties of the
    algorithm. Plotting only the best-HOTA trial removes that noise and
    keeps the comparison about the configuration each method actually
    selected.
    """
    has_fps = any(
        m.association_fps is not None
        for s in studies for m in s.trial_metrics.values()
    )
    if not has_fps:
        logger.info('No FPS data available. Skipping HOTA vs FPS chart.')
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for s in studies:
        m = s.trial_metrics.get(s.results.best_trial.number)
        if m is None or m.association_fps is None:
            continue
        ax.scatter(
            [m.association_fps], [m.hota],
            s=140, edgecolors='black', linewidths=1.2,
            marker='*', label=s.display_name, zorder=5,
        )

    ax.set_xlabel("Association FPS")
    ax.set_ylabel("HOTA")
    ax.set_title("HOTA vs FPS (best-HOTA trial per study)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

@dataclass
class GroupArtifacts:
    """Filenames produced for a single group's chart-set."""
    hota_chart_rel: str
    metrics_chart_rel: str
    fps_chart_rel: Optional[str]
    budget_chart_rel: Optional[str]
    walltime_chart_rel: Optional[str]


def _render_group_section(
    group_name: str,
    studies: List[StudyData],
    artifacts: GroupArtifacts,
    is_default: bool,
) -> str:
    heading = '## Summary' if is_default else f'## Group: `{group_name}`'
    lines: List[str] = [heading, '', build_summary_table(studies), '']
    lines.append(f'![Trial vs HOTA]({artifacts.hota_chart_rel})')
    lines.append('')
    lines.append(f'![Trial vs Metrics]({artifacts.metrics_chart_rel})')
    lines.append('')
    if artifacts.fps_chart_rel is not None:
        lines.append(f'![HOTA vs FPS]({artifacts.fps_chart_rel})')
        lines.append('')
    if artifacts.budget_chart_rel is not None:
        lines.append(f'![HOTA vs Budget]({artifacts.budget_chart_rel})')
        lines.append('')
    if artifacts.walltime_chart_rel is not None:
        lines.append(f'![HOTA vs Wall-time]({artifacts.walltime_chart_rel})')
        lines.append('')
    params_heading = '## Best Parameters' if is_default else f'### Best Parameters — `{group_name}`'
    lines += [params_heading, '', build_best_params_section(studies), '']
    return '\n'.join(lines)


def generate_report(
    grouped: List[Tuple[str, List[StudyData], GroupArtifacts]],
) -> str:
    """Build the Markdown report from per-group studies + chart artifacts."""
    lines: List[str] = ['# Optimization Comparison Report', '']
    is_single = len(grouped) == 1 and grouped[0][0] == DEFAULT_GROUP_NAME
    for group_name, studies, artifacts in grouped:
        lines.append(_render_group_section(group_name, studies, artifacts, is_default=is_single))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

def log_to_mlflow(
    cfg: GlobalConfig,
    studies: List[StudyData],
    artifact_paths: List[str],
) -> None:
    """Log report artifacts and per-study best metrics to MLflow."""
    if not is_mlflow_enabled(cfg.mlflow):
        return

    if cfg.mlflow.tracking_uri is not None:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    dataset_name = conventions.get_dataset_name(cfg.dataset.type, cfg.dataset.name)
    experiment_name = f'{dataset_name}/optimization-report/{cfg.inference.split}'
    mlflow.set_experiment(experiment_name)

    # Prefer the configured group keys for a stable, descriptive run name.
    groups_cfg = dict(getattr(cfg.report, 'groups', None) or {})
    if groups_cfg:
        candidate = 'report_' + '_'.join(groups_cfg.keys())
    else:
        display_names = [s.display_name for s in studies]
        joined = '_vs_'.join(display_names)
        candidate = f'report_{joined}'
        if len(candidate) > MAX_RUN_NAME_LEN:
            digest = hashlib.md5(joined.encode()).hexdigest()[:8]
            candidate = f'report_{display_names[0]}_and_{len(display_names) - 1}_more_{digest}'
    run_name = candidate[:MAX_RUN_NAME_LEN]

    with mlflow.start_run(run_name=run_name):
        for s in studies:
            prefix = s.display_name
            best = s.results.best_trial
            trials = s.completed_trials
            mlflow.log_metric(f'{prefix}/best_hota', best.value)
            mlflow.log_metric(f'{prefix}/best_trial', best.number)
            mlflow.log_metric(f'{prefix}/n_trials', len(trials))
            threshold = best.value * EFFICIENCY_RATIO
            trial_eff = first_trial_above(trials, threshold)
            if trial_eff is not None:
                mlflow.log_metric(f'{prefix}/trial_at_99pct', trial_eff)

            scenes_eff = first_budget_above(trials, threshold, 'scenes_evaluated')
            if scenes_eff is not None:
                mlflow.log_metric(f'{prefix}/scenes_at_99pct', scenes_eff)
            walltime_eff = first_budget_above(trials, threshold, 'wall_time_s')
            if walltime_eff is not None:
                mlflow.log_metric(f'{prefix}/walltime_at_99pct_s', walltime_eff)
            total_wt = total_wall_time(trials)
            if total_wt > 0:
                mlflow.log_metric(f'{prefix}/total_wall_time_s', total_wt)

            m = s.trial_metrics.get(best.number)
            if m is not None:
                if m.association_fps is not None:
                    mlflow.log_metric(f'{prefix}/best_assoc_fps', m.association_fps)
                if m.idf1 is not None:
                    mlflow.log_metric(f'{prefix}/best_idf1', m.idf1)
                if m.mota is not None:
                    mlflow.log_metric(f'{prefix}/best_mota', m.mota)
                if m.idsw is not None:
                    mlflow.log_metric(f'{prefix}/best_idsw', m.idsw)

        for path in artifact_paths:
            if os.path.exists(path):
                mlflow.log_artifact(path)

    logger.info(f'Logged MLflow run: experiment="{experiment_name}", run="{run_name}"')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _group_filename(basename: str, group_name: str) -> str:
    """Filename for a per-group chart; default group keeps the original name."""
    if group_name == DEFAULT_GROUP_NAME:
        return f'{basename}.png'
    return f'{basename}_{group_name}.png'


def _emit_group_charts(
    group_name: str,
    studies: List[StudyData],
    reports_dir: str,
) -> Tuple[GroupArtifacts, List[str]]:
    """Render every chart for one group, return artifacts + filesystem paths."""
    hota_filename = _group_filename(HOTA_CHART_BASENAME, group_name)
    metrics_filename = _group_filename(METRICS_CHART_BASENAME, group_name)
    fps_filename = _group_filename(FPS_CHART_BASENAME, group_name)
    budget_filename = _group_filename(BUDGET_CHART_BASENAME, group_name)
    walltime_filename = _group_filename(WALLTIME_CHART_BASENAME, group_name)

    hota_path = os.path.join(reports_dir, hota_filename)
    metrics_path = os.path.join(reports_dir, metrics_filename)
    fps_path = os.path.join(reports_dir, fps_filename)
    budget_path = os.path.join(reports_dir, budget_filename)
    walltime_path = os.path.join(reports_dir, walltime_filename)

    plot_trials(studies, hota_path)
    plot_trial_metrics(studies, metrics_path)
    plot_hota_vs_fps(studies, fps_path)
    plot_hota_vs_budget(studies, budget_path)
    plot_hota_vs_walltime(studies, walltime_path)

    artifacts = GroupArtifacts(
        hota_chart_rel=hota_filename,
        metrics_chart_rel=metrics_filename,
        fps_chart_rel=fps_filename if os.path.exists(fps_path) else None,
        budget_chart_rel=budget_filename if os.path.exists(budget_path) else None,
        walltime_chart_rel=walltime_filename if os.path.exists(walltime_path) else None,
    )

    fs_paths = [hota_path, metrics_path]
    for p in (fps_path, budget_path, walltime_path):
        if os.path.exists(p):
            fs_paths.append(p)
    return artifacts, fs_paths


def _run_report_inner(cfg: GlobalConfig) -> None:
    result_paths = discover_optimization_results(
        master_path=cfg.path.master,
        dataset_type=cfg.dataset.type,
        split=cfg.inference.split,
        dataset_name=cfg.dataset.name,
    )
    if not result_paths:
        logger.warning('No optimization results found. Nothing to report.')
        return

    logger.info(f'Found {len(result_paths)} optimization result(s).')
    studies = [load_study_data(p) for p in result_paths]

    reports_dir = conventions.get_reports_path(
        master_path=cfg.path.master,
        dataset_type=cfg.dataset.type,
        split=cfg.inference.split,
        dataset_name=cfg.dataset.name,
    )
    os.makedirs(reports_dir, exist_ok=True)

    groups_cfg: Dict[str, List[str]] = dict(getattr(cfg.report, 'groups', None) or {})
    grouped = _group_studies(studies, groups_cfg)
    if not grouped:
        logger.warning('No studies matched any configured group. Nothing to report.')
        return

    report_path = os.path.join(reports_dir, REPORT_FILENAME)
    fs_artifacts: List[str] = []
    rendered: List[Tuple[str, List[StudyData], GroupArtifacts]] = []
    for group_name, group_studies in grouped:
        artifacts, paths = _emit_group_charts(group_name, group_studies, reports_dir)
        rendered.append((group_name, group_studies, artifacts))
        fs_artifacts.extend(paths)

    report = generate_report(rendered)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f'Report written to {report_path}')

    log_to_mlflow(cfg, studies, artifact_paths=[report_path, *fs_artifacts])


@hydra.main(config_path=DANCETRACK_TRACKERS_CONFIG_PATH, config_name='optimization/report_per_family', version_base='1.1')
@pipeline.task('report_optimization')
def main(cfg: GlobalConfig) -> None:
    _run_report_inner(cfg)


if __name__ == '__main__':
    main()
