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
    frame_index: int,
    inplace: bool = True
  ) -> List[Tracklet]:
    ... Tracker logic ...

    return tracklets
```

Similarly, custom object detection inference, filter, association method
or dataset can also be implemented and seamlessly combined
with other components.

## Features
- **Tracker Algorithms Support**: 
  - SORT
  - ByteTrack
  - SparseTrack
- **Object Detection Inference**:
  - YOLOX
  - YOLOv8
- **Kalman Filter**:
  - Bot-Sort Kalman filter implementation
- **Association Methods**:
  - IoU (SORT)
  - Move
  - CBIoU
  - DCM
  - And more...
- **Dataset Format Support**:
  - MOT: MOT17, MOT20, DanceTrack 
- **Tools**:
  - Inference: Perform any tracker inference that can directly evaluated with TrackEval framework.
  - Postprocess: Perform offline postprocessing (linear interpolation, etc...) for more accuracy tracklets.
  - Visualize: Visualize tracker inference.

## Installation

Run these commands to install package within your virtual environment or docker container.

```bash
git clone https://github.com/Robotmurlock/Motrack
cd Motrack
pip install -e .
```
