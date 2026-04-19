# Implementation Notes

## Paper-Specified Elements

- Benchmark: ETH/UCY
- Time step: 0.4s (2.5Hz), observation 8, prediction 12
- Dynamic group reassignment at every observed timestep
- Grouping rules:
  - distance <= 1.0m
  - velocity difference <= 0.2m/s
  - directional coherence threshold `tau_o in [0.9, 1.0]`
- Hybrid graph: pedestrian nodes + group nodes
- Dynamic adjacency terms:
  - spatial distance
  - velocity term
  - directional alignment
  - conflict for group-group edges
  - row-wise normalization
- Enhanced attention with behavioural context vector
- Temporal convolution stack (7 layers)
- Group-aware VAE + GMM decoder (`M=3`)
- Deterministic prediction by highest mixture probability
- Training defaults:
  - Adam, lr=`1e-3`, weight decay=`1e-4`, batch=`64`, epochs=`500`
  - `lambda_nll=1.0`, `lambda_kl=0.1`

## Engineering Assumptions

1. **Grouping Partition Strategy**
   - The paper defines pairwise grouping criteria and disjoint groups.
   - Implementation uses connected components over a binary pairwise graph for deterministic group IDs.

2. **Intra-Consistency Equation Parsing**
   - PDF equation formatting is ambiguous around parenthesization.
   - Implementation follows a numerically stable interpretation aligned with intended behaviour:
     higher score for closer, similarly directed, and speed-similar members.

3. **Group Density Formula**
   - Paper requires density but does not fix a production formula.
   - Implementation uses a centroid/radial-distance proxy for stable relative density.

4. **Group Node Feature Conflict Scalar**
   - Pairwise conflict matrix is paper-specified.
   - For group node feature vector, a scalar summary is needed; implementation uses rowwise max normalized conflict.

5. **Temporal Summary for Decoder**
   - Decoder uses last temporal-convolution timestep embedding as per-agent temporal summary.

6. **Future Group Annotations in Export**
   - Predicted future group IDs/sizes are not directly specified in paper export design.
   - Predicted rows use default group placeholders in current exporter.

## Recommended Extensions

- Learn adjacency weights instead of fixed gamma coefficients
- Evaluate alternative group density definitions
- Add calibration metrics for multimodal outputs
- Add scene-wise ablation scripts for grouping thresholds
