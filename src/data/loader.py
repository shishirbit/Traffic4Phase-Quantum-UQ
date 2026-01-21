import os
from typing import Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import load_speed_npz, build_gaussian_adjacency_from_locations, adjacency_to_edge_index
from .dataset import temporal_split, fit_normalizer, SlidingWindowTrafficDataset

def load_data_and_graph(cfg: Dict[str, Any]):
    root = cfg["data"]["root"]
    npz_path = os.path.join(root, cfg["data"]["npz_file"])
    speed = load_speed_npz(npz_path, cfg["data"].get("array_key", "speed"))

    train_speed, val_speed, test_speed = temporal_split(
        speed,
        cfg["data"]["splits"]["train"],
        cfg["data"]["splits"]["val"],
        cfg["data"]["splits"]["test"]
    )

    normalizer = None
    if cfg["task"].get("normalize", True):
        normalizer = fit_normalizer(train_speed)
        train_speed = normalizer.transform(train_speed)
        val_speed = normalizer.transform(val_speed)
        test_speed = normalizer.transform(test_speed)

    # Adjacency
    adj_file = cfg["data"].get("adjacency_file", None)
    A = None
    if adj_file and os.path.exists(os.path.join(root, adj_file)):
        A = np.load(os.path.join(root, adj_file)).astype(np.float32)
    elif cfg["data"].get("build_adjacency", {}).get("enabled", False):
        loc_csv = os.path.join(root, cfg["data"]["locations_csv"])
        A = build_gaussian_adjacency_from_locations(
            loc_csv,
            sigma=cfg["data"]["build_adjacency"].get("sigma", "auto"),
            topk=cfg["data"]["build_adjacency"].get("topk", 20)
        )
    else:
        raise FileNotFoundError("No adjacency provided and adjacency building disabled.")

    edge_index_np, edge_weight_np = adjacency_to_edge_index(A, threshold=0.0)
    edge_index = torch.from_numpy(edge_index_np)
    edge_weight = torch.from_numpy(edge_weight_np)

    return (train_speed, val_speed, test_speed), normalizer, (edge_index, edge_weight), A

def make_loaders(cfg: Dict[str, Any], train_speed, val_speed, test_speed):
    Ts = cfg["task"]["history_steps"]
    H = cfg["task"]["horizon_steps"]
    stride = cfg["task"].get("stride", 1)
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"].get("num_workers", 0)

    train_ds = SlidingWindowTrafficDataset(train_speed, Ts, H, stride)
    val_ds = SlidingWindowTrafficDataset(val_speed, Ts, H, stride)
    test_ds = SlidingWindowTrafficDataset(test_speed, Ts, H, stride)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
