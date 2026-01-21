import os
import argparse
import time
import torch
import torch.nn as nn
from copy import deepcopy

from src.config import load_config
from src.utils import set_seed, ensure_dir, to_device
from src.data.loader import load_data_and_graph, make_loaders
from src.models.st_transformer import STTransformer
from src.losses import PinballLoss
from src.metrics import mae, rmse, coverage, avg_width

def mc_dropout_intervals(model, xb, passes: int, alpha: float):
    # Enable dropout masks
    model.enable_mc_dropout()
    preds = []
    for _ in range(passes):
        preds.append(model(xb))
    P = torch.stack(preds, dim=0)  # [T, B, H, N]
    lower = torch.quantile(P, alpha/2, dim=0)
    upper = torch.quantile(P, 1 - alpha/2, dim=0)
    median = torch.quantile(P, 0.5, dim=0)
    return lower, median, upper

def ensemble_intervals(models, xb, alpha: float):
    preds = [m(xb) for m in models]
    P = torch.stack(preds, dim=0)  # [M, B, H, N]
    lower = torch.quantile(P, alpha/2, dim=0)
    upper = torch.quantile(P, 1 - alpha/2, dim=0)
    median = torch.quantile(P, 0.5, dim=0)
    return lower, median, upper

def split_conformal(calib_lower, calib_upper, y_calib, alpha: float):
    # Nonconformity: max(lower - y, y - upper) (as in paper)
    r = torch.maximum(calib_lower - y_calib, y_calib - calib_upper)
    r = torch.clamp(r, min=0.0)
    q = torch.quantile(r.flatten(), 1 - alpha)
    return q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    (train_speed, val_speed, test_speed), normalizer, (edge_index, edge_weight), A = load_data_and_graph(cfg)
    train_loader, val_loader, test_loader = make_loaders(cfg, train_speed, val_speed, test_speed)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    # Base model for UQ: Phase 2 ST-Transformer with dropout
    p = cfg["phase2_transformer"]
    N = train_speed.shape[1]
    Ts = cfg["task"]["history_steps"]
    H = cfg["task"]["horizon_steps"]
    base = STTransformer(N, Ts, H, p["d_model"], p["nhead"], p["num_layers"], p["dim_feedforward"], p["dropout"]).to(device)

    # Load a trained checkpoint if available
    # (User can point to runs/.../best.pt by setting env CKPT)
    ckpt = os.environ.get("CKPT", None)
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        base.load_state_dict(state["model"], strict=False)
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print("No CKPT provided; intervals will be meaningless unless the model is trained.")

    alpha = float(cfg["uq"]["nominal_alpha"])
    passes = int(cfg["uq"]["mc_dropout"]["passes"])
    ens_M = int(cfg["uq"]["ensembles"]["members"])

    # Prepare ensemble members (simple: deepcopy base; in practice train independently)
    ensemble = [deepcopy(base).eval() for _ in range(ens_M)]

    # Conformal calibration set drawn from val loader (simple split)
    calib_batches = max(1, int(len(val_loader) * float(cfg["uq"]["conformal"]["calibration_fraction"])))
    calib_data = []
    for i, (xb, yb) in enumerate(val_loader):
        if i >= calib_batches:
            break
        calib_data.append((to_device(xb, device), to_device(yb, device)))

    # Compute conformal q using MC-dropout intervals (as an example)
    with torch.no_grad():
        rs = []
        for xb, yb in calib_data:
            l, m, u = mc_dropout_intervals(base, xb, passes, alpha)
            r = torch.maximum(l - yb, yb - u)
            r = torch.clamp(r, min=0.0)
            rs.append(r)
        r_all = torch.cat([r.flatten() for r in rs], dim=0)
        q = torch.quantile(r_all, 1 - alpha) if cfg["uq"]["conformal"]["enabled"] else torch.tensor(0.0, device=device)

    # Evaluate on test
    covs, widths, maes, rmses = [], [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = to_device(xb, device), to_device(yb, device)
            l, m, u = mc_dropout_intervals(base, xb, passes, alpha)
            l2, u2 = l - q, u + q
            covs.append(coverage(l2, u2, yb).cpu())
            widths.append(avg_width(l2, u2).cpu())
            maes.append(mae(m, yb).cpu())
            rmses.append(rmse(m, yb).cpu())

    print(f"MC-Dropout+Conformal: coverage={float(torch.stack(covs).mean()):.4f}, width={float(torch.stack(widths).mean()):.4f}")
    print(f"Point: MAE={float(torch.stack(maes).mean()):.4f}, RMSE={float(torch.stack(rmses).mean()):.4f}")

if __name__ == "__main__":
    main()
