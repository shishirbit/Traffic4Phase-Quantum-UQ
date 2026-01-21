import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class STGAT(nn.Module):
    """Phase 1: Per-time-step GAT followed by temporal Conv1D.
    Input: X [B, Ts, N]
    Output: Yhat [B, H, N]
    """
    def __init__(self, num_nodes: int, history_steps: int, horizon_steps: int,
                 gat_hidden: int = 16, gat_heads: int = 4, gat_dropout: float = 0.1,
                 temporal_conv_channels: int = 64, temporal_kernel: int = 3):
        super().__init__()
        self.N = num_nodes
        self.Ts = history_steps
        self.H = horizon_steps
        self.gat_hidden = gat_hidden
        self.gat_heads = gat_heads

        # GATConv expects node features [num_nodes_total, in_channels]
        self.gat = GATConv(
            in_channels=1,
            out_channels=gat_hidden,
            heads=gat_heads,
            concat=True,
            dropout=gat_dropout,
            add_self_loops=False
        )

        emb_dim = gat_hidden * gat_heads

        # Temporal convolution across time (Conv1d expects [B, C, L])
        self.temporal = nn.Conv1d(in_channels=emb_dim, out_channels=temporal_conv_channels,
                                 kernel_size=temporal_kernel, padding=temporal_kernel//2)
        self.act = nn.ReLU()
        self.head = nn.Linear(temporal_conv_channels, horizon_steps)

    def forward(self, x, edge_index, edge_weight=None):
        # x: [B, Ts, N]
        B, Ts, N = x.shape
        assert N == self.N and Ts == self.Ts

        # For each time step, apply GAT over graph
        gat_out = []
        for t in range(Ts):
            xt = x[:, t, :]  # [B, N]
            # Flatten batch into a big disconnected graph: edges repeated with offset
            # More efficient: loop over batch; simple reference version below.
            ht_list = []
            for b in range(B):
                node_feat = xt[b].unsqueeze(-1)  # [N, 1]
                ht = self.gat(node_feat, edge_index)  # [N, emb_dim]
                ht_list.append(ht)
            ht = torch.stack(ht_list, dim=0)  # [B, N, emb_dim]
            gat_out.append(ht)

        # Stack over time: [B, Ts, N, emb_dim] -> [B, N, emb_dim, Ts]
        Ht = torch.stack(gat_out, dim=1).permute(0, 2, 3, 1)
        # Apply temporal conv per node by merging B and N:
        Ht = Ht.reshape(B * N, Ht.shape[2], Ts)  # [B*N, emb_dim, Ts]
        z = self.act(self.temporal(Ht))          # [B*N, C, Ts]
        z = z.mean(dim=-1)                       # global average pooling -> [B*N, C]
        y = self.head(z)                         # [B*N, H]
        y = y.view(B, N, self.H).permute(0, 2, 1)  # [B, H, N]
        return y
