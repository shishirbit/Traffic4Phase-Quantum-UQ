import os
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-6)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + 1e-6) + self.mean

class SlidingWindowTrafficDataset(Dataset):
    def __init__(
        self,
        speed: np.ndarray,              # [T, N]
        history_steps: int,
        horizon_steps: int,
        stride: int = 1
    ):
        self.speed = speed
        self.T, self.N = speed.shape
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps
        self.stride = stride

        self.indices = []
        start = history_steps
        end = self.T - horizon_steps
        for t in range(start, end, stride):
            self.indices.append(t)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        x = self.speed[t - self.history_steps:t]          # [Ts, N]
        y = self.speed[t:t + self.horizon_steps]          # [H, N]
        # Return tensors as [Ts, N] and [H, N]
        return torch.from_numpy(x), torch.from_numpy(y)

def temporal_split(speed: np.ndarray, train_frac: float, val_frac: float, test_frac: float):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    T = speed.shape[0]
    t_train = int(T * train_frac)
    t_val = int(T * (train_frac + val_frac))
    train = speed[:t_train]
    val = speed[t_train:t_val]
    test = speed[t_val:]
    return train, val, test

def fit_normalizer(train_speed: np.ndarray) -> Normalizer:
    mean = train_speed.mean(axis=0, keepdims=True)   # [1, N]
    std = train_speed.std(axis=0, keepdims=True)     # [1, N]
    return Normalizer(mean=mean.astype(np.float32), std=std.astype(np.float32))
