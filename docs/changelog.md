# Changelog

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
