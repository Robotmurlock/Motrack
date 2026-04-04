"""
Validation script: compare motrack eval results against TrackEval.
Run with: uv run --with pytest scripts/validate_eval.py
"""
import sys
import os
import configparser

import numpy as np

# Add TrackEval to path
sys.path.insert(0, '/home/robotmurlock/Desktop/projects/TrackEval')

GT_FOLDER = '/media/home/DanceTrack-orig/val'
TRACKER_FOLDER = '/media/home/motrack-outputs/dancetrack/sort/val/2026-04-04_20-36-49_1c23ef076120/online'
SEQMAP_FILE = '/media/home/DanceTrack-orig/.seqmaps/val_seqmap.txt'


def get_scene_names_and_lengths():
    """Read scenes from seqmap and get lengths from seqinfo.ini."""
    import csv
    scenes = []
    lengths = {}
    with open(SEQMAP_FILE) as fp:
        reader = csv.reader(fp)
        for i, row in enumerate(reader):
            if i == 0 or row[0] == '':
                continue
            seq = row[0]
            scenes.append(seq)
            ini_file = os.path.join(GT_FOLDER, seq, 'seqinfo.ini')
            ini_data = configparser.ConfigParser()
            ini_data.read(ini_file)
            lengths[seq] = int(ini_data['Sequence']['seqLength'])
    return scenes, lengths


def run_trackeval():
    """Run TrackEval and return results."""
    import trackeval
    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 1,
        'PRINT_RESULTS': False,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': False,
        'OUTPUT_SUMMARY': False,
        'OUTPUT_DETAILED': False,
        'PLOT_CURVES': False,
    }
    dataset_config = {
        'GT_FOLDER': GT_FOLDER,
        'TRACKERS_FOLDER': TRACKER_FOLDER,
        'TRACKERS_TO_EVAL': [''],
        'TRACKER_SUB_FOLDER': '',
        'SPLIT_TO_EVAL': 'val',
        'SKIP_SPLIT_FOL': True,
        'SEQMAP_FILE': [SEQMAP_FILE],
        'PRINT_CONFIG': False,
    }
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [
        trackeval.metrics.HOTA({'PRINT_CONFIG': False}),
        trackeval.metrics.CLEAR({'PRINT_CONFIG': False}),
        trackeval.metrics.Identity({'PRINT_CONFIG': False}),
    ]
    output_res, _ = evaluator.evaluate([dataset], metrics)
    return output_res['MotChallenge2DBox']['']


def run_motrack_eval():
    """Run motrack eval and return results."""
    from motrack.eval import evaluate_tracker_output
    scenes, lengths = get_scene_names_and_lengths()
    return evaluate_tracker_output(
        gt_folder=GT_FOLDER,
        tracker_folder=TRACKER_FOLDER,
        scene_names=scenes,
        seq_lengths=lengths,
        eval_classes={1},
        distractor_classes={2, 7, 8, 12},
    )


def compare_results(te_results, mt_results):
    """Compare TrackEval vs motrack results."""
    te_combined = te_results['COMBINED_SEQ']['pedestrian']
    mt_combined = mt_results['combined']

    print("=" * 70)
    print("COMPARISON: TrackEval vs Motrack (combined)")
    print("=" * 70)

    # HOTA
    print("\n--- HOTA ---")
    for field in ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA']:
        te_val = np.mean(te_combined['HOTA'][field]) * 100
        mt_val = np.mean(mt_combined['HOTA'][field]) * 100
        diff = abs(te_val - mt_val)
        status = "OK" if diff < 0.01 else "MISMATCH"
        print(f"  {field:>10}: TE={te_val:8.4f}  MT={mt_val:8.4f}  diff={diff:.6f}  [{status}]")

    # CLEAR
    print("\n--- CLEAR ---")
    for field in ['MOTA', 'MOTP', 'CLR_Re', 'CLR_Pr']:
        te_val = float(te_combined['CLEAR'][field]) * 100
        mt_val = float(mt_combined['CLEAR'][field]) * 100
        diff = abs(te_val - mt_val)
        status = "OK" if diff < 0.01 else "MISMATCH"
        print(f"  {field:>10}: TE={te_val:8.4f}  MT={mt_val:8.4f}  diff={diff:.6f}  [{status}]")
    for field in ['CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'MT', 'PT', 'ML']:
        te_val = int(te_combined['CLEAR'][field])
        mt_val = int(mt_combined['CLEAR'][field])
        status = "OK" if te_val == mt_val else "MISMATCH"
        print(f"  {field:>10}: TE={te_val:8d}  MT={mt_val:8d}  [{status}]")

    # Identity
    print("\n--- Identity ---")
    for field in ['IDF1', 'IDR', 'IDP']:
        te_val = float(te_combined['Identity'][field]) * 100
        mt_val = float(mt_combined['Identity'][field]) * 100
        diff = abs(te_val - mt_val)
        status = "OK" if diff < 0.01 else "MISMATCH"
        print(f"  {field:>10}: TE={te_val:8.4f}  MT={mt_val:8.4f}  diff={diff:.6f}  [{status}]")
    for field in ['IDTP', 'IDFN', 'IDFP']:
        te_val = int(te_combined['Identity'][field])
        mt_val = int(mt_combined['Identity'][field])
        status = "OK" if te_val == mt_val else "MISMATCH"
        print(f"  {field:>10}: TE={te_val:8d}  MT={mt_val:8d}  [{status}]")


if __name__ == '__main__':
    print("Running TrackEval...")
    te_results = run_trackeval()
    print("Running motrack eval...")
    mt_results = run_motrack_eval()
    compare_results(te_results, mt_results)
