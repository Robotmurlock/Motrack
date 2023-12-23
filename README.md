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

### FastReID integration

Any [FastReID](https://github.com/JDAI-CV/fast-reid) model for appearance matching can be used.
Model has to be exported in ONNX. Please check [deploy documentation](https://github.com/JDAI-CV/fast-reid/tree/master/tools/deploy) for mode info.

### Supported datasets

Currently supported datasets are: MOT17, MOT20, DanceTrack.

Any custom dataset can be added by extending the base dataset.

### Tools

List of script tools:

  - Inference: Perform any tracker inference that can directly evaluated with TrackEval framework.
  - Postprocess: Perform offline postprocessing (linear interpolation, etc...) for more accuracy tracklets.
  - Visualize: Visualize tracker inference.

### Evaluation

Evaluation of different supported methods can be found [here](https://github.com/Robotmurlock/Motrack/blob/main/docs/evaluation.md).

## Installation

Run these commands to install package within your virtual environment or docker container.

```bash
pip install motrack
```

Package page can be found on [PyPI](https://pypi.org/project/motrack/).

### Extensions

In order to use `YOLOv8` for inference, please install `ultralytics` library:

```bash
pip install ultralytics
```

or install extras `motrack['yolov8']`:

```bash
pip install `motrack['yolov8']`
```

For `FastReID` inference, please install `onnxruntime` for CPU:

```bash
pip install onnxruntime
```

or GPU:

```bash
pip install onnxruntime-gpu
```

## Changelog

Package changelog can be found [here](https://github.com/Robotmurlock/Motrack/blob/main/docs/changelog.md)
