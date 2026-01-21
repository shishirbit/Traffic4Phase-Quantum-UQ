import torch
import torch.nn as nn
from .st_transformer import STTransformer
from .quantum import VariationalQuantumLayer

class QSTGTF(nn.Module):
    """Phase 3: Hybrid Quantum-Augmented ST-Transformer with simple fusion.

    Strategy:
    - Compute classical sensor embeddings from ST-Transformer internal last-step tokens
      (reuse STTransformer forward up to spatial attention input).
    - Create a pooled global vector (mean over sensors), feed into quantum layer
    - Concatenate quantum output to each sensor token and predict horizon.

    This is a practical approximation of the paper's description.
    """
    def __init__(self, num_nodes: int, history_steps: int, horizon_steps: int,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1,
                 nq: int = 8, q_layers: int = 4):
        super().__init__()
        self.base = STTransformer(num_nodes, history_steps, horizon_steps, d_model, nhead, num_layers, dim_feedforward, dropout)
        self.q = VariationalQuantumLayer(nq=nq, q_layers=q_layers)

        # Replace head: take [d_model + nq] -> H
        self.fuse = nn.Linear(d_model + nq, d_model)
        self.out = nn.Linear(d_model, horizon_steps)

    def forward(self, x):
        # Re-implement STTransformer forward to access tokens before final linear head
        B, Ts, N = x.shape
        xs = x.permute(0, 2, 1).contiguous().view(B * N, Ts, 1)
        e = self.base.in_proj(xs)
        e = self.base.pos(e)
        h = self.base.temporal_encoder(e)
        last = h[:, -1, :].view(B, N, self.base.d_model)
        tokens, _ = self.base.spatial_attn(last, last, last, need_weights=False)  # [B, N, d]

        pooled = tokens.mean(dim=1)  # [B, d]
        q_out = self.q(pooled)       # [B, nq]
        q_expand = q_out.unsqueeze(1).expand(-1, N, -1)  # [B, N, nq]

        fused = torch.cat([tokens, q_expand], dim=-1)    # [B, N, d+nq]
        fused = torch.relu(self.fuse(fused))             # [B, N, d]
        y = self.out(fused).permute(0, 2, 1).contiguous()  # [B, H, N]
        return y
