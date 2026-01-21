import os
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .metrics import mae, rmse
from .utils import ensure_dir, to_device

def train_loop(model: nn.Module, train_loader, val_loader, cfg: Dict[str, Any], run_dir: str, loss_fn: nn.Module):
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = Adam(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    sched = CosineAnnealingLR(opt, T_max=max(10, cfg["train"]["max_epochs"]))

    best_val = float("inf")
    patience = cfg["train"]["early_stop_patience"]
    bad = 0

    ensure_dir(run_dir)
    ckpt_path = os.path.join(run_dir, "best.pt")

    for epoch in range(cfg["train"]["max_epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}", leave=False)
        for xb, yb in pbar:
            xb, yb = to_device(xb, device), to_device(yb, device)
            opt.zero_grad()
            pred = model(xb) if not isinstance(model, tuple) else model[0](xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if cfg["train"].get("grad_clip", None):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))
        sched.step()

        # val
        model.eval()
        v_losses = []
        v_mae = []
        v_rmse = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = to_device(xb, device), to_device(yb, device)
                pred = model(xb)
                v_losses.append(loss_fn(pred, yb).detach().cpu())
                v_mae.append(mae(pred, yb).detach().cpu())
                v_rmse.append(rmse(pred, yb).detach().cpu())
        val_loss = float(torch.stack(v_losses).mean())
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad = 0
            torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                break

    # load best
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    return model, ckpt_path

@torch.no_grad()
def evaluate(model: nn.Module, loader, cfg: Dict[str, Any]):
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    maes, rmses = [], []
    for xb, yb in loader:
        xb, yb = to_device(xb, device), to_device(yb, device)
        pred = model(xb)
        maes.append(mae(pred, yb).cpu())
        rmses.append(rmse(pred, yb).cpu())
    return float(torch.stack(maes).mean()), float(torch.stack(rmses).mean())
