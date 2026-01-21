import torch

@torch.no_grad()
def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()

@torch.no_grad()
def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((pred - target) ** 2).mean())

@torch.no_grad()
def coverage(lower: torch.Tensor, upper: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Empirical coverage: fraction of y in [lower, upper]
    return ((y >= lower) & (y <= upper)).float().mean()

@torch.no_grad()
def avg_width(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return (upper - lower).mean()
