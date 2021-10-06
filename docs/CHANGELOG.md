## [0.2.0] - 2021-10-01

VFRAME is experimental, unstable and not yet recommended for production workflows. Version 0.1.0 improves processing of image groups and adds ONNX inference for GPU/CPU. More work is needed to improve usability for the CLI commands for selecting cpu/gpu-indexes.

### Changed
- removed batch modelzoo param type
- merged detect-batch and detect
- add experimental prehash option to calculate perceptual hashes in threaded video frame loader
- removed creating copy of frame for drawing unless requested
- GIF exporter to pipe processor
- save-* to save-json, save-images, save-video, save-gif commands
- file meta format to include date last modified, separates file meta and frame meta, application version
- to merge .env vars and modelzoo.yaml to single config.yaml (still use .env for password related variables)
- `vframe_cli` to `src`
- plugin location to `src/plugins`
- modelzoo yaml to use anchors and aliases 

### Added
- ONNX inference for YOLOV5 models
- PyTorch inference for YOLOV5 models
- Batch inference for PyTorch YOLOV5 models
- `pipe open` grouping for images, videos, and pre-processed items
- CHANGELOG to track changes

### Deprecated
- 2017-2018 cluster munition detector prototype models

### Fixed
- several small issues in drawing utils
- JPEG extension globbing in addition to JPG
- fixed `pipe open` command handling of image groups

### Removed
- .
