import torch
import torch.nn as nn
from .positional_encoding import SinusoidalPositionalEncoding

class STTransformer(nn.Module):
    """Phase 2: Temporal transformer per sensor, spatial attention across sensors.
    Input: X [B, Ts, N]
    Output: Yhat [B, H, N]
    """
    def __init__(self, num_nodes: int, history_steps: int, horizon_steps: int,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.N = num_nodes
        self.Ts = history_steps
        self.H = horizon_steps
        self.d_model = d_model

        # Input projection from scalar speed -> d_model
        self.in_proj = nn.Linear(1, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max(512, history_steps + 5))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="relu"
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Spatial attention at last time step: treat sensors as tokens
        self.spatial_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)

        self.out = nn.Linear(d_model, horizon_steps)

    def forward(self, x):
        # x: [B, Ts, N]
        B, Ts, N = x.shape
        assert N == self.N and Ts == self.Ts

        # Process each sensor independently through temporal transformer:
        # Reshape to [B*N, Ts, 1]
        xs = x.permute(0, 2, 1).contiguous().view(B * N, Ts, 1)
        e = self.in_proj(xs)          # [B*N, Ts, d]
        e = self.pos(e)
        h = self.temporal_encoder(e)  # [B*N, Ts, d]
        last = h[:, -1, :]            # [B*N, d]
        last = last.view(B, N, self.d_model)  # [B, N, d]

        # Spatial self-attention across sensors (tokens=N)
        attn_out, _ = self.spatial_attn(last, last, last, need_weights=False)  # [B, N, d]
        y = self.out(attn_out)  # [B, N, H]
        y = y.permute(0, 2, 1).contiguous()  # [B, H, N]
        return y

    def enable_mc_dropout(self):
        """Enable dropout during inference for MC-Dropout."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
