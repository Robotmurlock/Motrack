# Performance Profiling

Tracker FPS = harmonic mean (weights 0.5) of OD and ASSOC steps. Focus is on the ASSOC step.
Profiling scripts are standalone (synthetic data, no dataset needed): `python -m tools.profile.<script>`.

## Before/After Summary

|   T |   D | Before (ms) | After (ms) | Speedup |
|-----|-----|-------------|------------|---------|
|  30 |  10 |       0.715 |      0.413 |   1.73x |
|  30 |  20 |       0.889 |      0.452 |   1.97x |
|  30 |  30 |       1.065 |      0.544 |   1.96x |
|  60 |  10 |       1.378 |      0.762 |   1.81x |
|  60 |  20 |       1.716 |      0.834 |   2.06x |
|  60 |  30 |       2.080 |      0.978 |   2.13x |
| 100 | 100 |       5.268 |      2.641 |   2.00x |

**Component breakdown (T=100, D=100):**

| Component        | Before (ms) | After (ms) | Speedup |
|------------------|-------------|------------|---------|
| predict_all      | 1.676       | 1.013      | 1.65x   |
| form_cost_matrix | 3.081       | 1.153      | 2.67x   |
| hungarian        | 0.133       | 0.108      | -       |
| update_matched   | 0.374       | 0.363      | -       |
| **TOTAL**        | **5.268**   | **2.641**  | **2.0x** |

## Applied Optimizations

### 1. Vectorized IoU Cost Matrix

Replaced Python double loop in `form_cost_matrix()` with `BBox.batch_iou()` (numpy broadcasting).
Vectorized `IoUAssociation`, `HMIoUAssociation`, `DecayIoU` via `_calc_score_matrix()` override.
`AdaptiveIoU` keeps per-element fallback (history-dependent thresholds).

Raw IoU speedup: **69x** (PredBBox.iou 3.4ms vs numpy 0.05ms for 10k pairs).
End-to-end `form_cost_matrix` speedup: **2.67x** (remaining overhead: label gating loop, `as_numpy_xyxy` conversion).

### 2. Batch Kalman Predict

Added `batch_predict()` to filter interface, using existing `BotSortKalmanFilter.multi_predict()`.
Heavy matrix ops batched; per-tracklet project + PredBBox creation remains serial.

| N tracklets | Loop (ms) | Batch (ms) | Speedup |
|-------------|-----------|------------|---------|
| 10          | 0.088     | 0.032      | 2.79x   |
| 50          | 0.438     | 0.101      | 4.34x   |
| 100         | 0.889     | 0.185      | 4.81x   |

## End-to-End Breakdown (After)

| Component        | T=20,D=30 | T=50,D=50 | T=100,D=100 |
|------------------|-----------|-----------|-------------|
| predict_all      | 0.225     | 0.518     | 1.013       |
| form_cost_matrix | 0.108     | 0.320     | 1.153       |
| hungarian        | 0.020     | 0.041     | 0.108       |
| update_matched   | 0.221     | 0.117     | 0.363       |
| **TOTAL**        | **0.576** | **0.999** | **2.641**   |

## Not Optimized

- **Hungarian augmentation** — 2.5% of total frame time, marginal gain
- **OD cache** — 0.068ms/frame (3x npy), already faster than npz and mmap

## Tracker Benchmarks (cached OD)

**DanceTrack val (25508 frames):**

| Tracker     | Assoc FPS | E2E FPS |
|-------------|-----------|---------|
| SORT        | 1424      | 750     |
| ByteTrack   | 1245      | 660     |
| MoveSORT    | 909       | 532     |
| SparseTrack | 802       | 513     |

**SportsMOT val (26970 frames):**

| Tracker     | Assoc FPS | E2E FPS |
|-------------|-----------|---------|
| SORT        | 1150      | 529     |
| ByteTrack   | 994       | 527     |
| MoveSORT    | 848       | 493     |
| SparseTrack | 666       | 419     |
