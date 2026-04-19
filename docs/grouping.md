# Grouping

## Paper-Specified Components

At each observed timestep `t`, pedestrians `i` and `j` are grouped when:

- `||p_i^t - p_j^t|| <= 1.0m`
- `||v_i^t - v_j^t|| <= 0.2m/s`
- `o_i^t · o_j^t >= tau_o`, where `tau_o in [0.9, 1.0]`

Groups are recomputed independently at each timestep.

## Implemented Grouping Flow

1. Build pairwise binary grouping graph from rule checks
2. Extract connected components
3. Assign deterministic group IDs in sorted component order
4. Handle singletons as valid size-1 groups

## Group Feature Extraction

- **Intra-group consistency**
  - pairwise member consistency
  - group-level average consistency
- **Inter-group conflict**
  - directed pairwise group conflict based on geometry/velocity alignment
  - row-wise normalized conflict matrix
- **Group density**
  - centroid-based radial proxy
- **Group mean velocity**
  - per-group average velocity vector

Per-agent group-conditioned fields include:

- group ID
- group size
- intra-group consistency
- inter-group conflict score
- group density

## Paper-Specified vs Assumptions

- **Paper-specified**
  - threshold rules
  - timestep-wise dynamic reassignment
  - required group features
- **Engineering assumptions**
  - connected-components partition strategy
  - stable parsed form of intra-consistency equation
  - explicit density proxy formula
