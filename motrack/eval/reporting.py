"""
Evaluation result logging and JSON serialization.
"""
import json
import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger('EvalReporting')


def log_eval_results(results: Dict[str, Any], sequence_names: List[str]) -> None:
    """
    Logs per-sequence and combined evaluation results as a formatted table.
    """
    combined = results['combined']
    sequences = results['sequences']

    for metric_name, metric_res in combined.items():
        header = f'{metric_name} (combined)'
        fields = sorted(metric_res.keys())
        values = [_format_value(metric_res[f]) for f in fields]
        logger.info(f'{header}: ' + ', '.join(f'{f}={v}' for f, v in zip(fields, values)))

        for seq_name in sequence_names:
            if seq_name in sequences and metric_name in sequences[seq_name]:
                seq_res = sequences[seq_name][metric_name]
                seq_values = [_format_value(seq_res[f]) for f in fields]
                logger.info(f'  {seq_name}: ' + ', '.join(f'{f}={v}' for f, v in zip(fields, seq_values)))


def dump_eval_results_json(results: Dict[str, Any], output_path: str) -> None:
    """
    Serializes evaluation results to JSON.

    Numpy arrays are converted to their mean (for HOTA alpha-averaged fields)
    and all numpy scalars are converted to Python natives.
    """
    serializable = _to_serializable(results)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2)
    logger.info(f'Evaluation results saved to "{output_path}".')


def _format_value(v: Any) -> str:
    if isinstance(v, np.ndarray):
        return f'{100 * np.mean(v):.2f}'
    elif isinstance(v, float):
        return f'{100 * v:.2f}'
    elif isinstance(v, (np.floating, np.integer)):
        return f'{100 * float(v):.2f}'
    return str(v)


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return float(np.mean(obj))
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj
