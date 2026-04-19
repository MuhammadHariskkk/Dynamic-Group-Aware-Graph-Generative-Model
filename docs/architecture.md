# Architecture

## End-to-End Pipeline

1. **Preprocess ETH/UCY**
   - Parse trajectory files into fixed `(obs=8, pred=12)` windows
   - Save processed `.pt` artifacts with per-timestep motion states

2. **Dynamic Grouping**
   - At each observed timestep:
     - apply pairwise rule checks
     - partition via connected components
     - compute group features (consistency/conflict/density/velocity)

3. **Hybrid Graph**
   - Build timestep graph with node types:
     - pedestrians
     - groups
   - Build dynamic adjacency with weighted interaction terms

4. **Spatial Attention**
   - Compute socially informed attention with context vector:
     - distance
     - velocity difference
     - direction alignment
     - normalized conflict

5. **Temporal Convolution**
   - Process per-node attended features across observation window
   - Use 7-layer temporal stack

6. **Group-Aware Generative Prediction**
   - VAE encoder maps temporal + group context -> `(mu, logvar, z)`
   - GMM decoder maps `(z, temporal, context)` -> multimodal trajectories
   - Deterministic trajectory = highest mixture probability mode

7. **Training & Evaluation**
   - Loss: `L = lambda_nll * NLL + lambda_kl * KL`
   - Metrics: deterministic ADE/FDE, optional best-of-M ADE/FDE

8. **Export**
   - Structured CSV/NPY/NPZ outputs for ETH/UCY visualization workflows
   - Supports deterministic, multimodal, GT-only, combined exports

## Package Boundaries

- `groupaware.datasets`: preprocessing, dataset, collate, scene split
- `groupaware.grouping`: rules, partition, consistency/conflict/density, dynamic features
- `groupaware.graph`: node features, adjacency, hybrid graph, attention, temporal conv
- `groupaware.models`: group context, VAE encoder, GMM decoder, integrated model
- `groupaware.losses`: GMM NLL + KL
- `groupaware.metrics`: ADE/FDE deterministic + best-of-M
- `groupaware.trainers`: train/val loop, early stopping, checkpointing
- `groupaware.exporters`: schema + structured visualization export
- `groupaware.experiments`: train/eval/infer/export run orchestration
