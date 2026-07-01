import math

import torch
from torch import nn


class TrilEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Get the coordinates of the lower triangle
        i, j = torch.tril_indices(dim, dim)

        # Build a lookup grid: which index of the input belongs at (row, col)?
        grid = torch.zeros(dim, dim, dtype=torch.long)
        grid[i, j] = torch.arange(len(i))

        # Symmetrize the map: copy lower indices to upper
        grid = torch.maximum(grid, grid.T)

        # Store the flattened map as a buffer
        self.map = nn.Buffer(grid.flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expands x of shape (..., K) to (..., d, d),  where K = d * (d + 1) // 2
        """
        return x[..., self.map].view(*x.shape[:-1], self.dim, self.dim)


class KthEigval(nn.Module):
    def __init__(self, num_features, dim, k=None):
        super().__init__()
        self.lin = nn.Linear(num_features, dim * (dim + 1) // 2)
        self.tril_emb = TrilEmbed(dim)
        self.k = k or dim // 2

    def forward(self, x):
        mat_flat = self.lin(x)
        mat = self.tril_emb(mat_flat)
        eigvals = torch.linalg.eigvalsh(mat)
        return eigvals[..., self.k]


class KthEigval1DMonotone(nn.Module):
    def __init__(self, dim, k=None, alpha=0.1):
        super().__init__()
        self.bias_mat = nn.Parameter(torch.randn(dim * (dim + 1) // 2) / math.sqrt(dim))
        self.feat_vec = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
        self.tril_emb = TrilEmbed(dim)
        self.k = k or dim // 2

    def forward(self, x):
        feat_diag = nn.functional.softplus(self.feat_vec)
        mat = self.tril_emb(self.bias_mat)[None] + torch.diag_embed(
            x[:, None] * feat_diag
        )
        eigvals = torch.linalg.eigvalsh(mat)
        return eigvals[..., self.k]
