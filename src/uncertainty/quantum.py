from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

import pennylane as qml

@dataclass
class PCATransformer:
    components: np.ndarray  # [N, d]
    mean: np.ndarray        # [N,]
    n_components: int

    def transform(self, x: np.ndarray) -> np.ndarray:
        # x: [*, N]
        return (x - self.mean) @ self.components

class VariationalQuantumLayer(nn.Module):
    """Variational quantum circuit producing nq expectation values.

    Uses amplitude embedding of length 2^nq (pads/truncates input).
    Circuit: StronglyEntanglingLayers + PauliZ expectations.
    """
    def __init__(self, nq: int = 8, q_layers: int = 4, dev_name: str = "lightning.qubit"):
        super().__init__()
        self.nq = nq
        self.q_layers = q_layers
        self.dev = qml.device(dev_name, wires=nq)

        weight_shapes = {"weights": (q_layers, nq, 3)}
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):
            # inputs: [2^nq]
            qml.AmplitudeEmbedding(inputs, wires=range(nq), normalize=True, pad_with=0.0)
            qml.StronglyEntanglingLayers(weights, wires=range(nq))
            return [qml.expval(qml.PauliZ(i)) for i in range(nq)]

        self.circuit = circuit
        self.weights = nn.Parameter(0.01 * torch.randn(q_layers, nq, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d] (float)
        B, d = x.shape
        dim = 2 ** self.nq
        if d < dim:
            pad = torch.zeros(B, dim - d, device=x.device, dtype=x.dtype)
            inp = torch.cat([x, pad], dim=1)
        else:
            inp = x[:, :dim]
        outs = []
        for b in range(B):
            outs.append(torch.stack(self.circuit(inp[b], self.weights)))
        return torch.stack(outs, dim=0)  # [B, nq]
