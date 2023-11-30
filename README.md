# Motrack: Multi-Object Tracking Library

## Introduction

Motrack is a versatile multi-object tracking library designed to 
leverage the tracking-by-detection paradigm. 
It supports a range of tracker algorithms and object detections, 
making it ideal for applications in various domains.

## Features
- **Tracker Algorithms Support**: 
  - SORT
  - ByteTrack
  - SparseTrack
- **Object Detection Integration**:
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
