# Dynamic Group-Aware Graph Generative Model (ETH/UCY)

PyTorch implementation for:
**Dynamic Group-Aware Graph Generative Model for Socially Coherent Pedestrian Trajectory Prediction**.

Dataset scope is **ETH/UCY only** (ETH, Hotel, Univ, Zara1, Zara2).  


## What This Repository Implements

- Observation horizon `T_obs=8`, prediction horizon `T_pred=12`
- Dynamic group detection at each observed timestep
- Group feature extraction:
  - intra-group consistency
  - inter-group conflict (raw + row-wise normalized)
  - group mean velocity
  - group density
- Hybrid graph with pedestrian and group nodes
- Socially informed spatial attention with behavioural context `phi_ij`
- Seven-layer temporal convolution stack
- Group-aware VAE encoder + GMM decoder (`M=3`)
- Composite training loss: NLL + KL
- Deterministic evaluation from highest mixture probability mode
- Structured export for ETH/UCY visualization workflows, including compatibility target with:
  [InhwanBae/ETH-UCY-Trajectory-Visualizer](https://github.com/InhwanBae/ETH-UCY-Trajectory-Visualizer)

## Installation

```bash
python -m venv .venv
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .
```

## Configs

- Base: `configs/default.yaml`
- Scene overlays:
  - `configs/eth.yaml`
  - `configs/hotel.yaml`
  - `configs/univ.yaml`
  - `configs/zara1.yaml`
  - `configs/zara2.yaml`
- Loader utility: `groupaware.utils.config.load_config`

## Paper-Specified Hyperparameters Encoded

- Optimizer: Adam
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Batch size: `64`
- Epochs: `500`
- Latent dim: `5`
- MLP dims: `[16, 512, 8]`
- Temporal conv layers: `7`
- GMM modes: `3`
- Loss weights: `lambda_nll=1.0`, `lambda_kl=0.1`
- Grouping thresholds:
  - distance `<= 1.0 m`
  - velocity difference `<= 0.2 m/s`
  - directional coherence `tau_o in [0.9, 1.0]` (default `0.95`)
- Adjacency coefficients:
  - `gamma_spatial=0.25`
  - `gamma_vel=0.25`
  - `gamma_dir=0.25`
  - `gamma_conflict=0.25`

## Scientific Honesty

Ambiguous implementation details are explicitly documented as engineering assumptions in:

- `docs/implementation_notes.md`
- `docs/grouping.md`
- `docs/visualization_export.md`

No hidden assumptions are treated as paper facts.

## Data Preparation

Expected layout:

```text
data/
  raw/
    eth/
    hotel/
    univ/
    zara1/
    zara2/
  processed/
```

Preprocess:

```bash
python scripts/preprocess_data.py --config configs/default.yaml --scene all --raw-root data/raw --processed-root data/processed
```

## Training / Evaluation / Inference

Train:

```bash
python scripts/train.py --config configs/default.yaml --scene-config configs/eth.yaml --train-scene eth --val-scene hotel --device cpu
```

Evaluate all scenes:

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --scene all --device cpu
```

Infer:

```bash
python scripts/infer.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --scene all --device cpu
```

## Visualization Export (Primary Output Pathway)

Deterministic export (observed + GT + predictions):

```bash
python scripts/export_visualizer_data.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --scene all --include-observed --include-gt --run-name ethucy_det --format csv --device cpu
```

Multimodal export:

```bash
python scripts/export_visualizer_data.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --scene all --multimodal --include-observed --include-gt --run-name ethucy_mm --format npz --device cpu
```

Exports contain:

- `scene_id`, `sequence_id`, `agent_id`, `time_index`, `frame_index`
- `phase`, `mode_id`, `mode_probability`
- `x`, `y`, `vx`, `vy`, `heading_rad`
- `group_id`, `group_size`, `intra_group_consistency`, `inter_group_conflict`, `group_density`
- `is_prediction`, `is_deterministic`

Detailed mapping guidance: `docs/visualization_export.md`.

## Optional Debug Plots

```bash
python scripts/visualize_predictions.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --scene eth --batch-index 0 --sample-index 0 --show-groups --show-centroids --show-graph --show-conflict --plot-curves --device cpu
```

## Tests

```bash
pytest -q
```

## Repository Run Order

1. `scripts/preprocess_data.py`
2. `scripts/train.py`
3. `scripts/evaluate.py` and/or `scripts/infer.py`
4. `scripts/export_visualizer_data.py`
5. optional `scripts/visualize_predictions.py`

## Reproducibility Checklist

- Fixed `project.seed` in config
- Explicit ETH/UCY horizons (`8/12`)
- Deterministic group IDs per timestep
- Deterministic export row ordering
- Saved checkpoints (`best.pt`, `last.pt`)
- Saved history (`outputs/metrics/history.csv`, `history.json`)
