import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.spatial.distance import cdist

def load_speed_npz(npz_path: str, array_key: str = "speed") -> np.ndarray:
    obj = np.load(npz_path)
    if array_key not in obj:
        raise KeyError(f"Key '{array_key}' not found in {npz_path}. Available keys: {list(obj.keys())}")
    x = obj[array_key]
    if x.ndim != 2:
        raise ValueError(f"Expected speed array of shape [T, N], got {x.shape}")
    return x.astype(np.float32)

def build_gaussian_adjacency_from_locations(
    locations_csv: str,
    sigma: Optional[float] = None,
    topk: int = 20
) -> np.ndarray:
    df = pd.read_csv(locations_csv)
    # Accept common column names:
    lat_col = "latitude" if "latitude" in df.columns else "lat"
    lon_col = "longitude" if "longitude" in df.columns else "lon"
    coords = df[[lat_col, lon_col]].to_numpy(dtype=np.float64)
    # Euclidean in lat/lon is an approximation; for true geodesic use haversine.
    D = cdist(coords, coords, metric="euclidean")
    if sigma is None or sigma == "auto":
        sigma_val = D.mean()
    else:
        sigma_val = float(sigma)
    W = np.exp(-(D ** 2) / (2.0 * (sigma_val ** 2) + 1e-12))
    np.fill_diagonal(W, 0.0)
    # top-k sparsification per row
    if topk is not None and topk > 0:
        N = W.shape[0]
        W_sparse = np.zeros_like(W)
        for i in range(N):
            idx = np.argsort(W[i])[::-1][:topk]
            W_sparse[i, idx] = W[i, idx]
        W = W_sparse
    return W.astype(np.float32)

def adjacency_to_edge_index(A: np.ndarray, threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    # Returns edge_index (2, E) and edge_weight (E,)
    src, dst = np.where(A > threshold)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_weight = A[src, dst].astype(np.float32)
    return edge_index, edge_weight
