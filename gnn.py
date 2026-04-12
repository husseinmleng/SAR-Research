import torch
import torch.nn as nn
import torch.nn.functional as F


class AdjacencyModule(nn.Module):
    """Compute soft adjacency matrix from pairwise feature differences.

    For node features V of shape (B, N, D):
      phi_ij = |v_i - v_j|   shape (B, N, N, D)
    Apply 1x1 Conv MLP:
      Conv(D->64)/LReLU -> Conv(64->32)/LReLU -> Conv(32->1)
    Softmax over j (neighbors) -> A of shape (B, N, N).
    """

    def __init__(self, input_dim):
        super(AdjacencyModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, v):
        # v: (B, N, D)
        B, N, D = v.shape
        v_i = v.unsqueeze(2).expand(B, N, N, D)   # (B, N, N, D)
        v_j = v.unsqueeze(1).expand(B, N, N, D)   # (B, N, N, D)
        phi = torch.abs(v_i - v_j)                 # (B, N, N, D)

        phi = phi.permute(0, 3, 1, 2)              # (B, D, N, N)
        a = self.mlp(phi)                           # (B, 1, N, N)
        a = a.squeeze(1)                            # (B, N, N)
        a = F.softmax(a, dim=2)                    # normalize over neighbors
        return a                                    # (B, N, N)


class UpdateModule(nn.Module):
    """Weighted feature aggregation followed by a linear projection.

    V' = ReLU( FC( R @ V ) )   shape (B, N, hidden_dim)
    """

    def __init__(self, input_dim, hidden_dim=16):
        super(UpdateModule, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, a, v):
        # a: (B, N, N), v: (B, N, D)
        agg = torch.bmm(a, v)         # (B, N, D)
        out = F.relu(self.fc(agg))    # (B, N, H)
        return out


class GNN_module(nn.Module):
    """Paper GNN: 3 adjacency modules + 2 update modules with dense concat.

    Structure:
      R0 = Adj0(V0)
      V0' = Update0(R0, V0)
      V1 = cat(V0, V0')          dim grows D -> D+H
      R1 = Adj1(V1)
      V1' = Update1(R1, V1)
      V2 = cat(V1, V1')          dim grows D+H -> D+2H
      R2 = Adj2(V2)
      logits = FC( sum_j R2[:,0,j] * V2[:,j,:] )  -> (B, nway+1)
    """

    def __init__(self, nway, input_dim, hidden_dim=16, **kwargs):
        super(GNN_module, self).__init__()
        self.hidden_dim = hidden_dim

        dim0 = input_dim
        dim1 = dim0 + hidden_dim
        dim2 = dim1 + hidden_dim

        self.adj0 = AdjacencyModule(dim0)
        self.upd0 = UpdateModule(dim0, hidden_dim)

        self.adj1 = AdjacencyModule(dim1)
        self.upd1 = UpdateModule(dim1, hidden_dim)

        self.adj2 = AdjacencyModule(dim2)

        self.fc_out = nn.Linear(dim2, nway + 1)

    def forward(self, x):
        # x: (B, N, D)   N = nway+1 nodes (query first, then support)
        v = x

        # Layer 0
        r0 = self.adj0(v)                           # (B, N, N)
        v0_prime = self.upd0(r0, v)                 # (B, N, H)
        v = torch.cat([v, v0_prime], dim=2)         # (B, N, D+H)

        # Layer 1
        r1 = self.adj1(v)
        v1_prime = self.upd1(r1, v)
        v = torch.cat([v, v1_prime], dim=2)         # (B, N, D+2H)

        # Final adjacency for readout
        r2 = self.adj2(v)                           # (B, N, N)

        # Aggregate into query node (index 0) using R2[b, 0, :]
        w = r2[:, 0, :].unsqueeze(2)               # (B, N, 1)
        query_feat = (w * v).sum(dim=1)             # (B, D+2H)

        return self.fc_out(query_feat)              # (B, nway+1)
