# Source Layout

This directory is intentionally empty of implementation.

Simple structure:

- `src/app/`: entry point, argument parsing, dataset traversal, output writing
- `src/motion/`: feature extraction, matching, optical flow, temporal motion cues
- `src/localization/`: bounding box estimation from moving evidence
- `src/evaluation/`: IoU, detection accuracy, aggregation, summary export

## Ownership Rule

Prefer one member as the author for each source file.