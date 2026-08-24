"""
Collect the CMC sweep into one comparison table.

Each arm of the sweep is a separate experiment directory whose name ends in the arm's label,
so the results are gathered by globbing rather than by rerunning anything. Per-arm speed comes
from the inference run's own FPS stats: ``association_fps`` is the tracker plus CMC, which is
the number that separates the algorithms, while ``e2e_fps`` also carries detection and is
dominated by whether the detection cache was warm.

Usage:
    python -m tools.analysis.report_cmc --dataset mot17
    python -m tools.analysis.report_cmc --dataset mot20 --output docs/cmc/mot20.md
"""
import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

MASTER_TEMPLATE = '/media/home/motrack-outputs/{dataset}'

# MOT17 ran the full grid. MOT20 is a subset: there are no MOT20 GMC files, so the BoT-SORT
# reference arm cannot exist there, and the spatial-filter and kf model-order ablations were
# settled on MOT17 rather than repeated.
ARMS_BY_DATASET = {
    'mot17': [
        ('none', 'no CMC (baseline)'),
        ('gmc-file', '`gmc-from-file` (BoT-SORT reference)'),
        ('pylk', '`pylk` / shi-tomasi'),
        ('pylk-excl', '`pylk` / shi-tomasi + exclusion'),
        ('pylk-noransac', '`pylk` / shi-tomasi, RANSAC off'),
        ('orb', '`feature-matching` / ORB'),
        ('orb-excl', '`feature-matching` / ORB + exclusion'),
        ('sift', '`feature-matching` / SIFT'),
        ('sift-excl', '`feature-matching` / SIFT + exclusion'),
        ('pylk-cv', '`pylk` / shi-tomasi (OpenCV)'),
        ('pylk-excl-cv', '`pylk` / shi-tomasi + exclusion (OpenCV)'),
        ('orb-cv', '`feature-matching` / ORB (OpenCV)'),
        ('orb-excl-cv', '`feature-matching` / ORB + exclusion (OpenCV)'),
        ('sift-cv', '`feature-matching` / SIFT (OpenCV)'),
        ('sift-excl-cv', '`feature-matching` / SIFT + exclusion (OpenCV)'),
        ('orb-sfilter', '`feature-matching` / ORB + spatial filter'),
        ('orb-sfilter-std', '`feature-matching` / ORB + spatial filter (n_std)'),
        ('kf-center', '`kf-residual` / translation, centres'),
        ('kf-corners', '`kf-residual` / translation, corners'),
        ('kf-affine', '`kf-residual` / affine, corners'),
    ],
    'mot20': [
        ('none', 'no CMC (baseline)'),
        ('pylk', '`pylk` / shi-tomasi'),
        ('pylk-excl', '`pylk` / shi-tomasi + exclusion'),
        ('pylk-noransac', '`pylk` / shi-tomasi, RANSAC off'),
        ('orb', '`feature-matching` / ORB'),
        ('orb-excl', '`feature-matching` / ORB + exclusion'),
        ('orb-s1', '`feature-matching` / ORB, seed 1'),
        ('orb-s2', '`feature-matching` / ORB, seed 2'),
        ('sift', '`feature-matching` / SIFT'),
        ('sift-excl', '`feature-matching` / SIFT + exclusion'),
        ('pylk-cv', '`pylk` / shi-tomasi (OpenCV)'),
        ('pylk-excl-cv', '`pylk` / shi-tomasi + exclusion (OpenCV)'),
        ('orb-cv', '`feature-matching` / ORB (OpenCV)'),
        ('orb-excl-cv', '`feature-matching` / ORB + exclusion (OpenCV)'),
        ('sift-cv', '`feature-matching` / SIFT (OpenCV)'),
        ('sift-excl-cv', '`feature-matching` / SIFT + exclusion (OpenCV)'),
        ('kf-center', '`kf-residual` / translation, centres'),
    ],
}

# Set by main() from --dataset; module-level so the helpers stay simple.
MASTER = MASTER_TEMPLATE.format(dataset='mot17')
EXPERIMENT_PREFIX = 'mot17-sort-cmc-'
ARMS = ARMS_BY_DATASET['mot17']

# (group, name, label, is_ratio). TrackEval stores the quality metrics as ratios in [0, 1]
# and the event metrics as raw counts, so only the former are shown as percentages.
METRICS = [
    ('HOTA', 'HOTA', 'HOTA', True),
    ('HOTA', 'AssA', 'AssA', True),
    ('HOTA', 'DetA', 'DetA', True),
    ('Identity', 'IDF1', 'IDF1', True),
    ('CLEAR', 'MOTA', 'MOTA', True),
    ('CLEAR', 'IDSW', 'IDSW', False),
    ('CLEAR', 'Frag', 'Frag', False),
]


# Cross-tracker study: does the effect measured on SORT carry to other trackers? SORT's arms
# were run first under the main sweep's naming, so its slugs are mapped rather than renamed.
CROSS_TRACKERS = [
    ('sort', 'SORT', {'none': 'none', 'kf': 'kf-center', 'orbcap': 'orb-sfilter',
                      'siftexcl': 'sift-excl', 'gmcfile': 'gmc-file'}),
    ('bytetrack', 'ByteTrack', {}),
    ('movesort', 'MoveSORT', {}),
    ('sparsetrack', 'SparseTrack', {}),
]

CROSS_ARMS = [
    ('none', 'no CMC'),
    ('kf', 'MR-CMC'),
    ('orbcap', 'ORB + cap'),
    ('siftexcl', 'SIFT + masking'),
    ('gmcfile', '`gmc-from-file`'),
]


def find_experiment_dir(experiment: str) -> Optional[str]:
    """
    Locates a run directory by full experiment name.
    """
    pattern = os.path.join(MASTER, experiment, 'val', 'inference', '*', 'eval_results.json')
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    return os.path.dirname(matches[-1]) if matches else None


def cross_hota(tracker: str, arm: str, slugs: dict) -> Optional[float]:
    """
    Combined HOTA for one tracker/arm cell, or None when it has not been run.
    """
    if tracker == 'sort':
        experiment = f'{EXPERIMENT_PREFIX}{slugs[arm]}'
    else:
        experiment = f'mot17-{tracker}-cmc-{arm}'

    run_dir = find_experiment_dir(experiment)
    if run_dir is None:
        return None

    with open(os.path.join(run_dir, 'eval_results.json'), 'r', encoding='utf-8') as handle:
        results = json.load(handle)
    value = results.get('combined', {}).get('HOTA', {}).get('HOTA')
    return float(value) * 100 if value is not None else None


def format_cross() -> str:
    """
    Renders the tracker x CMC matrix, with each cell's delta against that tracker's own baseline.

    The delta matters more than the absolute here: trackers differ in baseline strength, so the
    question is whether CMC buys the same thing on each, not which tracker scores highest.
    """
    lines = ['| tracker | ' + ' | '.join(label for _, label in CROSS_ARMS) + ' |',
             '|---' * (len(CROSS_ARMS) + 1) + '|']

    for tracker, label, slugs in CROSS_TRACKERS:
        baseline = cross_hota(tracker, 'none', slugs)
        cells = []
        for arm, _ in CROSS_ARMS:
            value = cross_hota(tracker, arm, slugs)
            if value is None:
                cells.append('—')
            elif arm == 'none' or baseline is None:
                cells.append(f'{value:.2f}')
            else:
                cells.append(f'{value:.2f} ({value - baseline:+.2f})')
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')

    return '\n'.join(lines)


def find_run_dir(arm: str) -> Optional[str]:
    """
    Locates an arm's run directory.

    The directory name is a hash of the config, so it is discovered rather than constructed.
    A config change produces a new hash and therefore a new directory, which is why more than
    one may exist; the most recent wins.
    """
    pattern = os.path.join(MASTER, EXPERIMENT_PREFIX + arm, 'val', 'inference', '*', 'eval_results.json')
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    return os.path.dirname(matches[-1]) if matches else None


def read_arm(arm: str) -> Optional[dict]:
    """
    Reads one arm's evaluation results and speed, or None when it has not been run.
    """
    run_dir = find_run_dir(arm)
    if run_dir is None:
        return None

    with open(os.path.join(run_dir, 'eval_results.json'), 'r', encoding='utf-8') as handle:
        results = json.load(handle)

    association_fps = None
    fps_path = os.path.join(run_dir, 'fps_stats.json')
    if os.path.exists(fps_path):
        with open(fps_path, 'r', encoding='utf-8') as handle:
            stats = json.load(handle)
        association_fps = stats.get('association_fps')

    return {'results': results, 'association_fps': association_fps, 'run_dir': run_dir}


def metric(entry: dict, group: str, name: str) -> Optional[float]:
    """
    Pulls one combined metric, tolerating a metric group the evaluator did not emit.
    """
    value = entry['results'].get('combined', {}).get(group, {}).get(name)
    return float(value) if value is not None else None


def format_table(rows: List[Tuple[str, dict]]) -> str:
    """
    Renders the comparison as a Markdown table, with each arm's delta against the baseline.
    """
    baseline = next((entry for arm, entry in rows if arm == 'none'), None)
    baseline_hota = metric(baseline, 'HOTA', 'HOTA') if baseline else None

    header = '| variant | ' + ' | '.join(label for _, _, label, _ in METRICS) + ' | ΔHOTA | assoc FPS |'
    separator = '|---' * (len(METRICS) + 3) + '|'
    lines = [header, separator]

    for arm, entry in rows:
        label = dict(ARMS)[arm]
        if entry is None:
            lines.append(f'| {label} | ' + ' | '.join(['—'] * (len(METRICS) + 2)) + ' |')
            continue

        cells = []
        for group, name, _, is_ratio in METRICS:
            value = metric(entry, group, name)
            if value is None:
                cells.append('—')
            else:
                cells.append(f'{value * 100:.2f}' if is_ratio else f'{value:.0f}')

        hota = metric(entry, 'HOTA', 'HOTA')
        if baseline_hota is None or hota is None or arm == 'none':
            delta = '—'
        else:
            delta = f'{(hota - baseline_hota) * 100:+.2f}'

        fps = entry['association_fps']
        cells.append(delta)
        cells.append('—' if fps is None else f'{fps:.1f}')
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')

    return '\n'.join(lines)


def format_per_sequence(rows: List[Tuple[str, dict]], dataset: str = 'mot17') -> str:
    """
    Renders per-sequence HOTA, which is where the static/moving camera split shows up.

    A CMC method can only help where the camera actually moves, so a combined number alone
    hides whether an improvement came from the sequences it was supposed to come from.
    """
    scenes: List[str] = []
    for _, entry in rows:
        if entry is not None:
            scenes = sorted(entry['results'].get('sequences', {}))
            break
    if not scenes:
        return ''

    # A moving camera is the only condition CMC can act on, so the two groups are marked.
    # MOT20 is filmed entirely from static cameras, so it has no moving group at all.
    moving = {'mot17': {'MOT17-05', 'MOT17-10', 'MOT17-11', 'MOT17-13'}, 'mot20': set()}[dataset]
    short = [scene.replace('-FRCNN-H2', '').replace('-H2', '') for scene in scenes]
    labels = [f'{name} *' if name in moving else name for name in short]
    lines = ['| variant | ' + ' | '.join(labels) + ' |', '|---' * (len(labels) + 1) + '|']

    for arm, entry in rows:
        label = dict(ARMS)[arm]
        if entry is None:
            lines.append(f'| {label} | ' + ' | '.join(['—'] * len(scenes)) + ' |')
            continue
        cells = []
        for scene in scenes:
            value = entry['results']['sequences'].get(scene, {}).get('HOTA', {}).get('HOTA')
            cells.append('—' if value is None else f'{float(value) * 100:.2f}')
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')

    lines.append('')
    if moving:
        lines.append('`*` moving camera. The static sequences are the control: CMC has nothing to compensate there.')
    else:
        lines.append('All sequences are filmed from static cameras, so every column is a control.')

    return '\n'.join(lines)


def main() -> None:
    global MASTER, EXPERIMENT_PREFIX, ARMS  # pylint: disable=global-statement

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=sorted(ARMS_BY_DATASET), default='mot17')
    parser.add_argument('--output', type=str, default=None, help='Write the tables to this Markdown file')
    args = parser.parse_args()

    MASTER = MASTER_TEMPLATE.format(dataset=args.dataset)
    EXPERIMENT_PREFIX = f'{args.dataset}-sort-cmc-'
    ARMS = ARMS_BY_DATASET[args.dataset]

    rows = [(arm, read_arm(arm)) for arm, _ in ARMS]

    missing = [arm for arm, entry in rows if entry is None]
    if missing:
        print(f'warning: no results found for {", ".join(missing)}')

    n_scenes = {'mot17': 7, 'mot20': 4}[args.dataset]
    blocks = [
        f'### Combined over the {n_scenes} {args.dataset.upper()} val sequences',
        format_table(rows),
        '### Per-sequence HOTA',
        format_per_sequence(rows, dataset=args.dataset),
    ]
    # The cross-tracker study was run on MOT17 only.
    if args.dataset == 'mot17':
        blocks += [
            '### Cross-tracker (HOTA, delta against that tracker\'s own no-CMC baseline)',
            format_cross(),
        ]
    report = '\n\n'.join(blocks)

    print(report)
    if args.output is not None:
        with open(args.output, 'w', encoding='utf-8') as handle:
            handle.write(report + '\n')
        print(f'\nwritten to {args.output}')


if __name__ == '__main__':
    main()
