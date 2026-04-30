# Motrack: Multi-Object Tracking Library

## Introduction

Motrack is a versatile multi-object tracking library designed to 
leverage the tracking-by-detection paradigm. 
It supports a range of tracker algorithms and object detections, 
making it ideal for applications in various domains.

## Usage

Pseudocode for tracker utilization:

```python
from motrack.object_detection import YOLOv8Inference
from motrack.tracker import ByteTracker, TrackletState

tracker = ByteTracker()  # Default parameters
tracklets = []
yolo = YOLOv8Inference(...)

video_frames = read_video(video_path)

for i, image in enumerate(video_frames):
  detections = yolo.predict_bboxes(image)
  tracklets = tracker.track(tracklets, detections, i)
  active_tracklets = [t for t in tracklets if t.state == TrackletState.ACTIVE]

  foo_bar(active_tracklets)
```

This library offers flexibility to use any custom object detector.

Implementation of custom tracker:

```python
from typing import List, Tuple

from motrack.library.cv.bbox import PredBBox
from motrack.tracker import Tracker, Tracklet


class MyTracker(Tracker):
  def track(
    self,
    tracklets: List[Tracklet],
    detections: List[PredBBox],
    frame_index: int
  ) -> List[Tracklet]:
    ... Tracker logic ...

    return tracklets
```

Similarly, custom object detection inference, filter, association method
or dataset can also be implemented and seamlessly combined
with other components.

## Features

### Supported tracker algorithms

| Method Name | Description                                            |
|-------------|--------------------------------------------------------|
| SORT        | [arxiv: Simple Online and Realtime Tracking](https://arxiv.org/pdf/1602.00763.pdf)    | 
| DeepSORT    | [arxiv: SIMPLE ONLINE AND REALTIME TRACKING WITH A DEEP ASSOCIATION METRIC](https://arxiv.org/pdf/1703.07402.pdf) |
| MoveSORT    | SORT with improved association method                  |
| ByteTrack   | [arxiv: ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)   |
| Bot-SORT    | [arxiv: BoT-SORT: Robust Associations Multi-Pedestrian Tracking](https://arxiv.org/abs/2206.14651)    |
| SparseTrack | [arxiv: SparseTrack: Multi-Object Tracking by Performing Scene Decomposition based on Pseudo-Depth](https://arxiv.org/abs/2306.05238) |

Evaluation of these methods on different datasets can be found in [evaluation.md](https://github.com/Robotmurlock/Motrack/blob/main/docs/evaluation.md)

### Supported object detection algorithms

| Method Name | Description                                                                        |
|-------------|------------------------------------------------------------------------------------|
| YOLOX       | [arxiv: Simple Online and Realtime Tracking](https://arxiv.org/pdf/1602.00763.pdf) | 
| YOLOv8      | [github: Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)                             |

Use `tools/detection/create_yolo_format.py` to create a YOLOv8 training dataset and
`tools/detection/create_coco_format.py` to create a YOLOX training dataset.

### FastReID integration

Any [FastReID](https://github.com/JDAI-CV/fast-reid) model for appearance matching can be used.
Model has to be exported in ONNX. Please check [deploy documentation](https://github.com/JDAI-CV/fast-reid/tree/master/tools/deploy) for mode info.
Use `tools/reid/create_fastreid_patches.py` to create a FastReID dataset for training an appearance model.

### Supported datasets

Currently supported datasets are: MOT17, MOT20, DanceTrack and SportsMOT.

Any custom dataset can be added by extending the base dataset.

### Tools

After installation, three console scripts are available from any directory:

  - **`motrack-inference`**: Run tracker inference. Outputs are stored under a deterministic hash-based directory.
  - **`motrack-eval`**: Evaluate tracker outputs with HOTA, CLEAR, Identity, and Count metrics. Results are logged and saved as JSON.
  - **`motrack-optimize`**: Run Optuna hyperparameter optimization over a tracker's search space.

Equivalent invocations:

```bash
uv run motrack-inference                   # console script via uv
uv run python -m motrack.cli.inference     # module run via uv
python tools/inference.py                  # legacy (still works)
```

If your venv is already activated (`source .venv/bin/activate`), drop the `uv run` prefix.

Two additional tools are available as scripts in the repo:

  - **Postprocess** (`tools/postprocess.py`): Offline postprocessing (linear interpolation, etc.).
  - **Visualize** (`tools/visualize_inference.py`): Visualize tracker inference.

### Library API

The CLI is a thin Hydra wrapper around the library API. Anything the CLI does can be called directly:

```python
from motrack.tools import run_inference, run_eval, run_optimize

run_inference(cfg)                   # build dataset/detector/tracker and run
results = run_eval(cfg)              # HOTA / CLEAR / Identity / Count
run_optimize(cfg)                    # full Optuna study
```

Dataset construction is pluggable — pass a custom builder if your library wraps a different dataset layer:

```python
from motrack.tools import run_optimize, DatasetBuilder
from motrack.datasets import BaseDataset

def my_dataset_builder(cfg) -> BaseDataset:
    return MyCustomDataset(cfg.dataset.params)

run_optimize(cfg, dataset_builder=my_dataset_builder)
```

### Evaluation

Motrack includes a built-in evaluation module (`motrack/eval`) that computes
HOTA, CLEAR (MOTA/MOTP), Identity (IDF1), and Count metrics — no external
TrackEval installation required. Run evaluation after inference:

```bash
uv run motrack-eval
```

Results are saved as `eval_results.json` inside the tracker run directory.

Evaluation of different supported methods can be found [here](https://github.com/Robotmurlock/Motrack/blob/main/docs/evaluation.md).

## Installation

Motrack is developed and tested with [uv](https://docs.astral.sh/uv/). Adding it to a uv-managed project:

```bash
uv add motrack
```

Or installing into the current environment:

```bash
uv pip install motrack
```

Pip works too if you prefer:

```bash
pip install motrack
```

Package page can be found on [PyPI](https://pypi.org/project/motrack/).

### Working from a clone

```bash
git clone https://github.com/Robotmurlock/Motrack.git
cd Motrack
uv sync
```

This installs all runtime dependencies and registers the `motrack-inference`, `motrack-eval`, and `motrack-optimize` console scripts in the project venv.

### Optional extras

| Extra | Provides |
|---|---|
| `yolov8` | `ultralytics` (YOLOv8 inference) |
| `reid`   | `onnxruntime` (FastReID inference) |
| `optuna` | `optuna` (hyperparameter search) |
| `mlflow` | `mlflow` (experiment tracking) |

Install one or more:

```bash
uv add 'motrack[yolov8,reid]'
# or
uv pip install 'motrack[yolov8,reid]'
# or (pip)
pip install 'motrack[yolov8,reid]'
```

For GPU `onnxruntime`, replace the `reid` extra with a direct install:

```bash
uv add onnxruntime-gpu     # or: pip install onnxruntime-gpu
```

## Changelog

Package changelog can be found [here](https://github.com/Robotmurlock/Motrack/blob/main/docs/changelog.md)
