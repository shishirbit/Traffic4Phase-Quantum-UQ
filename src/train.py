import os
import argparse
import time
import torch
import torch.nn as nn

from src.config import load_config
from src.utils import set_seed, ensure_dir
from src.data.loader import load_data_and_graph, make_loaders
from src.models.st_gat import STGAT
from src.models.st_transformer import STTransformer
from src.models.q_stgtf import QSTGTF
from src.trainer import train_loop, evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, choices=[1,2,3])
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--run_dir", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    (train_speed, val_speed, test_speed), normalizer, (edge_index, edge_weight), A = load_data_and_graph(cfg)
    train_loader, val_loader, test_loader = make_loaders(cfg, train_speed, val_speed, test_speed)

    N = train_speed.shape[1]
    Ts = cfg["task"]["history_steps"]
    H = cfg["task"]["horizon_steps"]

    if args.phase == 1:
        p = cfg["phase1_stgat"]
        model = STGAT(
            num_nodes=N, history_steps=Ts, horizon_steps=H,
            gat_hidden=p["gat_hidden"], gat_heads=p["gat_heads"], gat_dropout=p["gat_dropout"],
            temporal_conv_channels=p["temporal_conv_channels"], temporal_kernel=p["temporal_kernel"]
        )
        # Wrap forward to include graph
        class Wrapper(nn.Module):
            def __init__(self, m, edge_index, edge_weight):
                super().__init__()
                self.m = m
                self.register_buffer("edge_index", edge_index)
                self.register_buffer("edge_weight", edge_weight)
            def forward(self, x):
                return self.m(x, self.edge_index, self.edge_weight)
        model = Wrapper(model, edge_index, edge_weight)

        loss_fn = nn.SmoothL1Loss()  # Huber
        tag = "phase1_stgat"

    elif args.phase == 2:
        p = cfg["phase2_transformer"]
        model = STTransformer(
            num_nodes=N, history_steps=Ts, horizon_steps=H,
            d_model=p["d_model"], nhead=p["nhead"], num_layers=p["num_layers"],
            dim_feedforward=p["dim_feedforward"], dropout=p["dropout"]
        )
        loss_fn = nn.MSELoss()
        tag = "phase2_sttransformer"

    else:
        p = cfg["phase2_transformer"]
        q = cfg["phase3_quantum"]
        model = QSTGTF(
            num_nodes=N, history_steps=Ts, horizon_steps=H,
            d_model=p["d_model"], nhead=p["nhead"], num_layers=p["num_layers"],
            dim_feedforward=p["dim_feedforward"], dropout=p["dropout"],
            nq=q["qubits"], q_layers=q["q_layers"]
        )
        loss_fn = nn.MSELoss()
        tag = "phase3_qstgtf"

    run_dir = args.run_dir or os.path.join("runs", tag + "_" + time.strftime("%Y%m%d_%H%M%S"))
    ensure_dir(run_dir)

    model, ckpt = train_loop(model, train_loader, val_loader, cfg, run_dir, loss_fn)
    test_mae, test_rmse = evaluate(model, test_loader, cfg)
    print(f"Saved: {ckpt}")
    print(f"Test MAE: {test_mae:.4f} | Test RMSE: {test_rmse:.4f}")

if __name__ == "__main__":
    main()
