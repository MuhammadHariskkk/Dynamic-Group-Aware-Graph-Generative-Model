# Visualization Export

## Goal

Primary output pathway is structured export for ETH/UCY workflows, with compatibility target toward:
[ETH-UCY-Trajectory-Visualizer](https://github.com/InhwanBae/ETH-UCY-Trajectory-Visualizer)

Matplotlib plotting is optional debug-only support.

## Supported Export Modes

- deterministic predictions
- multimodal predictions (all modes + probabilities)
- observed + ground-truth only
- combined export (observed + GT + predictions)

## File Policy

- formats: `csv`, `npy`, `npz`
- deterministic row ordering
- one-file-per-scene and/or one-file-per-run

## Export Schema

Required columns:

- `scene_id`
- `sequence_id`
- `agent_id`
- `time_index`
- `frame_index`
- `phase` (`observed`, `ground_truth`, `predicted`)
- `mode_id`
- `mode_probability`
- `x`, `y`
- `vx`, `vy`
- `heading_rad`
- `group_id`
- `group_size`
- `intra_group_consistency`
- `inter_group_conflict`
- `group_density`
- `is_prediction`
- `is_deterministic`

## Mapping Notes for External Visualizers

- Scene and sequence identifiers remain explicit
- Per-point coordinates and velocities are included
- Mode probabilities are explicit for multimodal exports
- If an external visualizer expects a narrower schema, adapter mapping can be done by:
  - filtering by `phase`
  - selecting deterministic rows (`is_deterministic=1`) or all modes
  - renaming subset columns

## Paper-Specified vs Assumptions

- **Paper-specified**
  - deterministic selection by highest mode probability for evaluation
  - multimodal trajectory distribution output
- **Engineering assumptions**
  - predicted rows use placeholder future group fields where future group inference is not explicitly defined by paper export spec
  - adapter-friendly flat table schema for robust interoperability
