# Changelog

## 0.8.0 - 2026-05-23

### Features
- Added **Multi-Fidelity Greedy Coordinate Search (MFGCS)** optimizer (`motrack.optimization.mfgcs`): one-axis-at-a-time coordinate descent evaluated on a small scene subset, full-eval acceptance gate, shrinking-radius windows, barren-sweep early stop, and per-trial budget metrics (`scenes_evaluated`, `wall_time_s`)
- Added **HyperbandPruner** rung schedule to the Optuna pipeline: a `pruner:` block on `RandomParams` / `TPEParams` switches the objective into rung-based multi-fidelity evaluation; trials report HOTA at each rung and weak trials are killed at low fidelity
- Added **GP / WarmGP** samplers (`gp` / `warm_gp` pipelines) backed by Optuna's `GPSampler`, with the manually-tuned default enqueued as the warm-start trial for `warm_gp`
- Added **StratifiedSceneSampler** for MFGCS: picks `n_per_group` scenes from each regex-defined group, yielding a balanced low-fidelity subset on multi-domain datasets (e.g. SportsMOT basketball / football / volleyball)
- Added a grouped **optimization report** (`tools/analysis/report_optimization.py`): per-family, cross-tracker, and best-across-families groupings driven by Hydra config; HOTA vs trial / budget / wall-time / association-FPS plots; cumulative-budget summaries logged to MLflow
- Added an **optional `scenes` filter** on `run_eval` so an evaluation can be restricted to a subset of scenes (used by MFGCS at the rung gate)
- Exposed `fuse_score` on every IoU / move / DCM matcher and `duplicate_iou_threshold` on SparseTrack, making them tunable in the new MFGCS family configs
- Migrated the MLflow stack to **Postgres + MinIO** (`docker/docker-compose.yaml`), replacing the SQLite + local-fs backend. New services: `motrack-mlflow`, `motrack-mlflow-postgres`, `motrack-mlflow-minio`
- Added SportsMOT optimization configs (per-tracker TPE / MFGCS variants plus `_v2` / `_n10` ablations and a cross-tracker report grouping)
- Added the `tools/migrate/backfill_pre_c2_budget.py` one-off tool to reconstruct `scenes_evaluated` / `wall_time_s` on pre-C2 trial records (and optionally re-log them to MLflow)

### Refactor
- Converted `motrack.tools.optimization` into a package and split the HPO *library* (samplers, pipelines, results) from the *driver* (CLI / Hydra glue) so external callers can build a study without going through Hydra
- Unified sampler instantiation behind a factory keyed by `optimizer.sampler` so adding a new sampler is one entry, not one branch per call site
- Restructured the DanceTrack optimization configs into an `optimization/` subdirectory with `# @package _global_` and absolute `/...` defaults so Hydra resolves them against the project root regardless of the entry-point's location
- Aligned the MFGCS eval cache with full-coverage subsets: when `scene_sampler.n == |D|`, the rung-eval is short-circuited to a full-eval cache hit, making `n = |D|` behave as plain coordinate descent without a redundant gate eval

### Docs
- Added `docs/optimization/report.md` with the final HPO write-up comparing TPE / Random / GP / MFGCS families on DanceTrack and SportsMOT, plus 25 figures

## 0.7.0 - 2026-05-01

### Features
- Added mmdetection-based YOLOX inference with ByteTrack checkpoint weight remapping
- Migrated from `setup.py` to `pyproject.toml` with `uv` package management
- Split Docker into mmdet (`Dockerfile`) and legacy YOLOX (`yolox.Dockerfile`) images
- Added centralized filesystem conventions for tracker outputs with dataset-level naming and deterministic run hashes
- Added integrated evaluation module (`motrack/eval`) with HOTA, CLEAR, Identity, and Count metrics — replaces external TrackEval CLI dependency
- Added `tools/eval.py` entrypoint for evaluating tracker outputs with JSON result export
- Added configurable eval/distractor class IDs per dataset for evaluation preprocessing
- Added Optuna-based hyperparameter optimization with TPE / warm-start TPE / random samplers, dependent search-space parameters (`min_param` / `max_param`), and config-hash-based caching across trials
- Added MLflow experiment tracking integration (optional via `motrack[mlflow]` extra)
- Exposed inference / eval / optimize as a library API: `motrack.tools.run_inference`, `motrack.tools.run_eval`, `motrack.tools.run_optimize` are importable and usable from external tracker libraries
- Added pluggable dataset construction via `motrack.tools.DatasetBuilder` — pass a custom builder to `run_inference` / `run_eval` / `run_optimize` to integrate datasets that aren't in `dataset_factory`
- Added `motrack.cli` subpackage with thin Hydra wrappers; registered `motrack-inference`, `motrack-eval`, `motrack-optimize` as console scripts via `[project.scripts]` (runnable from any directory after install)
- Promoted result schemas to public API: `motrack.eval.results.EvalResults`, `motrack.tools.{InferenceOutputData, OptunaOutputData, OptimizationResults, TrialResult, ExperimentResults, TrackerRunResult}`

### Refactor
- Restructured configs into `trackers/`, `od/`, and `deprecated/` standalone directories
- Lazy YOLOX imports to avoid hard dependency when using mmdet
- Renamed executable entrypoints from `scripts/` to `tools/`
- Renamed tracker output directories from `active` / `all` / `postprocess` to `online` / `debug` / `offline`
- Renamed `motrack/evaluation` package to `motrack/inference` (IO module)
- Renamed `TrackerEvalConfig` to `TrackerInferenceConfig` and config group `eval` to `inference` (the previous name conflicted with the new evaluation module)
- Switched run directory naming from `{datetime}_{hash}` to hash-only for deterministic path lookup
- Lifted orchestration logic out of `tools/` into the `motrack` package so it can be imported by external libraries; `tools/{inference,eval,optimize}.py` are now 1-line forwarders to `motrack.cli.*`
- Removed `tools/data/`; its contents moved to their natural homes in the package (`motrack/eval/results.py`, `motrack/tools/inference.py`, `motrack/tools/optimization.py`, `motrack/tools/results.py`)

### Fixes
- Fixed pandas `drop()` compatibility with newer versions
- Added explicit dataset output names to tracker and deprecated configs so MOT-family datasets no longer share the same `mot` result directory
- Fixed stale `tools.data` imports in `motrack.eval.reporting` and `motrack.tools.mlflow_logger` that would have broken any external import of those modules

## 0.6.0 - 2026-03-29

### Features
- Tracklet can now store frame data in its history
- Kalman filter adaptive parameter can be set to true/false
- Pixel density analysis script

### Experimental
- Motion models with image features

## 0.4.1 - 2024-02-26

### Features
- Added script for `FastReID` training dataset creation
- Added script for `YOLOv8` training dataset creation
- Added script for `YOLOX` training dataset creation

## 0.4.0 - 2024-01-31

### Features
- Added support for SportsMOT dataset (evaluation still not added)
- ByteTrack now supports ReID on low confidence detections
- Motrack-motion filter models now support CMC

## 0.3.1 - 2024-01-11

### Features
- Implementation of HVC (improved Move) association method
- Implementation of LongTermReID association method
- Tracking postprocess now includes minimum tracklet length
- Generalized Motrack-motion package support (usage of any motrack-motion filter)
  - This currently includes the RNNFilter and TransFilter methods
- Implementation of OC_SORT's observation centric momentum association methods
- Implementation od DTIoU (decaying threshold) IoU based association method

### Fixes
- ByteTrack lost tracklets are properly extrapolated with a filter

## 0.3.0 - 2023-12-31

### Features
- Extension of motion filters with Motrack-motion library (End-to-end RNNFilter)
- Implementation of Hybrid-SORT's HMIoU

### Fixes
- Visualization now shows track id properly and does not crash

## 0.2.2 - 2023-12-29

### Fixes
- Appearance embedding update (improved SORT-ReID evaluation score)

## 0.2.1 - 2023-12-24

### Features
- Implementation of weighted cost matrix composition for easy heuristic combination
- Implementation of KF for confidence modeling
- Implementation of Hybrid-SORT inspired confidence association method
- Implement Dockerfile and docker-compose

### Fixes
- Remove Byte low detections ReID inference for faster inference
- Remove Pytorch Lightning config print dependency

### Docs
- Add results for SORT with ReID
- Add results for MoveByte with confidence modeling
- Separate custom and standard method results

## 0.2.0 - 2023-12-23

### Features
- Support for custom FastReID with ONNX export
- Implementation of SORT-ReID algorithm (DeepSORT-like tracker with modern ReID algorithms)

## 0.1.1 - 2023-12-15

### Features
- Support for custom CMC algorithm, with GMC from file for evaluation on popular datasets
- Support for Bot-SORT, SparseTrack with GMC, FastTracker (no-motion filter with greedy association)

### Docs
- Added `evaluation.md` with algorithms evaluated on DanceTrack

### Fixes
- ByteTrack lost new tracks.

## 0.1.0 - 2023-12-09

### Features
- Support for custom datasets, tracker algorithms, association algorithms, object detection algorithms or motion filters
- Support SORT, MoveSORT, ByteTrack, SparseTrack (without CMC)
