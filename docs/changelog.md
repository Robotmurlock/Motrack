# Changelog

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
