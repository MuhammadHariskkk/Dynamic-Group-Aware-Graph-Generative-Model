# Group Rule Reference

## Paper-specified thresholds

- Distance threshold: `<= 1.0 m`
- Velocity difference threshold: `<= 0.2 m/s`
- Directional coherence threshold: `tau_o` in `[0.9, 1.0]`

## Default repository value

- `tau_o = 0.95` (configurable per scene/config)

## Engineering assumption for group formation

When pairwise criteria are satisfied, an undirected edge is created in a per-timestep pedestrian graph.
Groups are derived as connected components. This makes grouping deterministic and implementation-safe.

## Planned implementation outputs

- Pedestrian-to-group mapping per timestep
- Group-level statistics:
  - mean velocity
  - density
  - intra-group consistency
  - inter-group conflict
