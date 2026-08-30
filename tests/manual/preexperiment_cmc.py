"""
Pre-experiment: how do the CMC variants behave on real footage?

Runs each variant over consecutive frame pairs and reports intrinsic measures, without
running a tracker. Tracking metrics come later; the point here is to catch anything obviously
wrong before spending a sweep on it, and to see whether the variants agree with each other and
with the published reference.

Three things are measured:

- **Apparent motion per sequence.** MOT17 val is a mix of static and moving camera sequences.
  A static sequence should produce a near-identity warp; anything else is a bug, not a result.
  MOT20 val is expected to be static throughout, which is the premise its benchmark's control
  reading rests on, so it is measured rather than assumed.
- **Agreement with `gmc-from-file`.** BoT-SORT's precomputed warps are the closest thing to
  ground truth available here. They are not truth - they are another estimate - so a
  disagreement locates a difference rather than proving anyone wrong. MOT17 only: BoT-SORT
  published no MOT20 warps.
- **Cost and failure rate.** Runtime per frame, and how often a variant gives up and returns
  identity. A variant that falls back on a third of frames is not measuring what its name says.

Detections for the exclusion variants come from ground truth rather than a detector, which
upper-bounds what exclusion can achieve: any effect seen here is the best case, before
detector recall enters.

Run with: uv run python tests/manual/preexperiment_cmc.py --dataset {mot17,mot20}
"""
import argparse
import configparser
import glob
import os
import time
from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np

from motrack.cmc.algorithms.base import CMCContext
from motrack.cmc.factory import cmc_factory
from motrack.library.cv.bbox import BBox, PredBBox

GMC_DIR = '/media/home/models/cmc/mot17_gmc'
TARGET_LONG_EDGE = 960

# MOT17-05 is 640x480 at 14 fps; the rest are 1920x1080 at 25-30 fps.
MOT17_SCENES = {
    'MOT17-02-FRCNN-H2': 'static',
    'MOT17-04-FRCNN-H2': 'static',
    'MOT17-09-FRCNN-H2': 'static',
    'MOT17-05-FRCNN-H2': 'moving',
    'MOT17-10-FRCNN-H2': 'moving',
    'MOT17-11-FRCNN-H2': 'moving',
    'MOT17-13-FRCNN-H2': 'moving',
}

# MOT20 is filmed from fixed cameras. That is the premise the benchmark's control reading rests
# on, so it is measured here rather than taken from the dataset description.
MOT20_SCENES = {
    'MOT20-01-H2': 'expected static',
    'MOT20-02-H2': 'expected static',
    'MOT20-03-H2': 'expected static',
    'MOT20-05-H2': 'expected static',
}

DATASETS = {
    'mot17': ('/media/home/MOT17-orig/val', MOT17_SCENES),
    'mot20': ('/media/home/MOT20-orig/val', MOT20_SCENES),
}

# Set by main(); module-level so the loaders stay simple.
MOT17_ROOT, SCENES = DATASETS['mot17']

VARIANTS = {
    'pylk/shi-tomasi': ('pylk', {}),
    'pylk/shi-tomasi +excl': ('pylk', {'exclusion': {'enabled': True}}),
    'match/orb': ('feature-matching', {}),
    'match/orb +excl': ('feature-matching', {'exclusion': {'enabled': True}}),
    'match/sift': ('feature-matching', {'feature_detector': {'type': 'sift'}}),
    'match/sift +excl': ('feature-matching', {'feature_detector': {'type': 'sift'}, 'exclusion': {'enabled': True}}),
}


def native_size(scene: str) -> tuple:
    """
    Reads the sequence's native frame size from `seqinfo.ini`.

    The reference warps are stored in native pixels, so they must be normalized by the native
    size rather than by whatever resolution the algorithms happen to run at.
    """
    parser = configparser.ConfigParser()
    parser.read(os.path.join(MOT17_ROOT, scene, 'seqinfo.ini'))
    return int(parser['Sequence']['imwidth']), int(parser['Sequence']['imheight'])


def load_frames(scene: str, n_frames: int) -> List[np.ndarray]:
    """
    Loads and resizes the first `n_frames` images of a scene.
    """
    paths = sorted(glob.glob(os.path.join(MOT17_ROOT, scene, 'img1', '*.jpg')))[:n_frames]
    frames = []
    for path in paths:
        image = cv2.imread(path)
        scale = TARGET_LONG_EDGE / max(image.shape[:2])
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return frames


def load_detections(scene: str, n_frames: int) -> Dict[int, List[PredBBox]]:
    """
    Loads ground-truth boxes per frame, normalized, as a stand-in for detections.

    MOT17 columns are `frame, id, x, y, w, h, consider, class, visibility`. Only pedestrians
    that count towards evaluation are kept, so the boxes match what a detector is asked to find.
    """
    width, height = native_size(scene)
    per_frame = defaultdict(list)

    with open(os.path.join(MOT17_ROOT, scene, 'gt', 'gt.txt'), 'r', encoding='utf-8') as handle:
        for line in handle:
            fields = line.strip().split(',')
            frame_id = int(fields[0])
            if frame_id > n_frames or int(fields[6]) != 1 or int(fields[7]) != 1:
                continue
            x, y, w, h = (float(v) for v in fields[2:6])
            bbox = BBox.from_xyxy(x / width, y / height, (x + w) / width, (y + h) / height, clip=True)
            # Frame ids are one-based on disk and zero-based inside the tracker.
            per_frame[frame_id - 1].append(PredBBox.create(bbox=bbox, label=0, conf=1.0))

    return per_frame


def translation_magnitude(warp: np.ndarray) -> float:
    """
    Length of the warp's translation, in normalized units.
    """
    return float(np.hypot(warp[0, 2], warp[1, 2]))


def is_identity(warp: np.ndarray) -> bool:
    return bool(np.allclose(warp, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32), atol=1e-9))


def run_scene(scene: str, n_frames: int) -> Dict[str, dict]:
    """
    Runs every variant plus the reference over one scene.

    Returns:
        Per-variant statistics
    """
    frames = load_frames(scene, n_frames)
    detections = load_detections(scene, n_frames)
    width, height = frames[0].shape[1], frames[0].shape[0]

    # BoT-SORT published GMC files for MOT17 only, so the reference column is MOT17-only too.
    reference_warps = {}
    if scene.startswith('MOT17'):
        reference = cmc_factory('gmc-from-file', {'dirpath': GMC_DIR})
        reference.reset()
        for index in range(1, len(frames)):
            # The stored warps are in native pixels, so they are normalized by the native size.
            reference_warps[index] = reference.apply(
                CMCContext(frame_index=index, scene=scene, image_size=native_size(scene))
            )

    results = {}
    for name, (key, params) in VARIANTS.items():
        cmc = cmc_factory(key, params)
        cmc.reset()

        magnitudes, disagreements, fallbacks = [], [], 0
        start = time.perf_counter()
        for index in range(1, len(frames)):
            warp = cmc.apply(CMCContext(
                frame_index=index,
                scene=scene,
                prev_frame=frames[index - 1],
                curr_frame=frames[index],
                image_size=(width, height),
                detections=detections.get(index, [])
            ))
            magnitudes.append(translation_magnitude(warp))
            fallbacks += int(is_identity(warp))
            if index in reference_warps:
                disagreements.append(float(np.linalg.norm(warp[:, 2] - reference_warps[index][:, 2])))
        elapsed = (time.perf_counter() - start) * 1000.0 / max(len(frames) - 1, 1)

        results[name] = {
            'magnitude': float(np.median(magnitudes)),
            'disagreement': float(np.median(disagreements)) if disagreements else float('nan'),
            'fallback_rate': fallbacks / max(len(frames) - 1, 1),
            'ms_per_frame': elapsed,
        }

    results['gmc-from-file'] = {
        'magnitude': float(np.median([translation_magnitude(w) for w in reference_warps.values()]))
        if reference_warps else float('nan'),
        'disagreement': 0.0,
        'fallback_rate': 0.0,
        'ms_per_frame': 0.0,
    }
    # How much of the frame the detections cover bounds what exclusion can possibly change.
    coverage = [sum(d.area for d in detections.get(i, [])) for i in range(1, len(frames))]
    results['coverage'] = float(np.median(coverage)) if coverage else 0.0
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=sorted(DATASETS), default='mot17')
    parser.add_argument('--frames', type=int, default=30, help='Consecutive frames per scene')
    parser.add_argument('--scenes', type=str, nargs='*', default=None, help='Scenes to run')
    args = parser.parse_args()

    global MOT17_ROOT, SCENES  # pylint: disable=global-statement
    MOT17_ROOT, SCENES = DATASETS[args.dataset]
    if args.scenes is None:
        args.scenes = list(SCENES)

    per_scene = {}
    for scene in args.scenes:
        print(f'running {scene} ({SCENES.get(scene, "?")}) ...', flush=True)
        per_scene[scene] = run_scene(scene, args.frames)

    names = list(VARIANTS) + ['gmc-from-file']

    print()
    print(f'=== median |translation| in normalized units, {args.frames - 1} frame pairs per scene')
    print(f'{"scene":24}{"camera":9}{"cover":>7}' + ''.join(f'{n:>24}' for n in names))
    for scene, results in per_scene.items():
        row = ''.join(f'{results[n]["magnitude"]:24.6f}' for n in names)
        print(f'{scene:24}{SCENES.get(scene, "?"):9}{results["coverage"] * 100:6.1f}%' + row)

    print()
    print('=== median disagreement with gmc-from-file (normalized translation distance)')
    print(f'{"scene":24}{"camera":9}' + ''.join(f'{n:>24}' for n in names[:-1]))
    for scene, results in per_scene.items():
        row = ''.join(f'{results[n]["disagreement"]:24.6f}' for n in names[:-1])
        print(f'{scene:24}{SCENES.get(scene, "?"):9}' + row)

    print()
    print('=== identity fallback rate / cost')
    print(f'{"variant":26}{"fallback":>12}{"ms/frame":>12}')
    for name in names[:-1]:
        fallback = np.mean([per_scene[s][name]['fallback_rate'] for s in per_scene])
        cost = np.mean([per_scene[s][name]['ms_per_frame'] for s in per_scene])
        print(f'{name:26}{fallback * 100:11.1f}%{cost:12.1f}')


if __name__ == '__main__':
    main()
