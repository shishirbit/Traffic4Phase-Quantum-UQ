import torch
import torch.nn as nn
import torch.nn.functional as F

class PinballLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        # preds: [B, H, N, Q]
        # target: [B, H, N]
        losses = []
        for qi, q in enumerate(self.quantiles):
            e = target - preds[..., qi]
            losses.append(torch.max((q - 1) * e, q * e).mean())
        return sum(losses) / len(losses)
