# Changelog

## 0.2.0 - 2023-12-23

### Added
- Support for custom FastReID with ONNX export
- Implementation of SORT-ReID algorithm (DeepSORT-like tracker with modern ReID algorithms)

## 0.1.1 - 2023-12-15

### Added
- Support for custom CMC algorithm, with GMC from file for evaluation on popular datasets
- Support for Bot-SORT, SparseTrack with GMC, FastTracker (no-motion filter with greedy association)

### Docs
- Added `evaluation.md` with algorithms evaluated on DanceTrack

### Fixed
- ByteTrack lost new tracks.

## 0.1.0 - 2023-12-09

### Added
- Support for custom datasets, tracker algorithms, association algorithms, object detection algorithms or motion filters
- Support SORT, MoveSORT, ByteTrack, SparseTrack (without CMC)
