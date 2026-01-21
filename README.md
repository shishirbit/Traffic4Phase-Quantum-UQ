# Four-Phase Spatio-Temporal + Quantum-Enhanced Traffic Speed Forecasting (Reference Implementation)

This repository is a **reproducible reference implementation** of the framework described in:
"A Four-Phase Spatio-Temporal and Quantum-Enhanced Framework for Traffic Speed Forecasting with Calibrated Uncertainty"
(IEEE Access-style manuscript). It implements:

- **Phase 1**: ST-GAT (Graph Attention per time step + temporal Conv1D)
- **Phase 2**: ST-Transformer (temporal self-attention per node + spatial attention)
- **Phase 3**: Q-STGTF (hybrid classical Transformer + variational quantum feature mixing)
- **Phase 4**: Uncertainty suite (MC-Dropout, Ensembles, Quantile Regression, Split-Conformal)

> Note: The METR-LA dataset is not redistributed here. This repo expects standard METR-LA format used by DCRNN/STGCN baselines.

## 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Torch Geometric notes
Torch Geometric wheels depend on your CUDA/Torch version; see the official installation page if needed.

## 2) Data layout (expected)

Place METR-LA files under:

```
data/METR-LA/
  metr-la.npz            # speed tensor, shape [T, N]
  sensor_locations.csv   # columns: sensor_id, latitude, longitude (optional if adjacency provided)
  adjacency.npy          # optional, shape [N, N]
```

If `adjacency.npy` is not provided but locations are, the repo can build a Gaussian-kernel adjacency (top-k sparsified).

## 3) Run training

### Phase 1: ST-GAT
```bash
python -m src.train --phase 1 --config configs/default.yaml
```

### Phase 2: ST-Transformer
```bash
python -m src.train --phase 2 --config configs/default.yaml
```

### Phase 3: Hybrid Quantum (Q-STGTF)
```bash
python -m src.train --phase 3 --config configs/default.yaml
```

### Phase 4: Uncertainty + Conformal
```bash
python -m src.uncertainty.run_uq --config configs/default.yaml
```

## 4) Reproducing paper metrics

- Point metrics: MAE / RMSE (median or mean predictions)
- UQ metrics: empirical coverage and average width for nominal 95% intervals
- Conformal: split-conformal calibration on a held-out calibration set

Reproducibility controls:
- Deterministic seeds
- Fixed temporal split order (no leakage)
- Saved configs and checkpoints under `runs/`

## 5) Applying to new datasets

Provide a `*.npz` file with array `speed` of shape `[T, N]` and optionally sensor coordinates.
Update `configs/default.yaml` accordingly.

## License
MIT (suggested).
