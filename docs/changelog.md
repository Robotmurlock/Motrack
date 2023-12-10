# Changelog

## 0.1.1 - Unreleased

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
