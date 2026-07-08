from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

type ModelKind = Literal["unconstrained", "monotone"]


class TrilEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        i, j = torch.tril_indices(dim, dim)

        grid = torch.zeros(dim, dim, dtype=torch.long)
        grid[i, j] = torch.arange(len(i))
        grid = torch.maximum(grid, grid.T)

        self.register_buffer("map", grid.flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand lower-triangular coordinates to symmetric matrices."""
        return x[..., self.map].view(*x.shape[:-1], self.dim, self.dim)


class KthEigval(nn.Module):
    def __init__(self, num_features: int, dim: int, eig_idx: int | None = None):
        super().__init__()
        self.dim = dim
        self.eig_idx = dim // 2 if eig_idx is None else eig_idx
        if not 0 <= self.eig_idx < dim:
            raise ValueError(f"eig_idx must be in [0, {dim}); got {self.eig_idx}")

        self.lin = nn.Linear(num_features, dim * (dim + 1) // 2)
        self.tril_emb = TrilEmbed(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mat = self.tril_emb(self.lin(x))
        eigvals = torch.linalg.eigvalsh(mat)
        return eigvals[..., self.eig_idx]


class KthEigval1DMonotone(nn.Module):
    """
    f(x) = lambda_k(A0 + x diag(p)), p >= 0.

    eig_idx is zero-based.
    """

    def __init__(self, dim: int, eig_idx: int | None = None):
        super().__init__()
        self.dim = dim
        self.eig_idx = dim // 2 if eig_idx is None else eig_idx
        if not 0 <= self.eig_idx < dim:
            raise ValueError(f"eig_idx must be in [0, {dim}); got {self.eig_idx}")

        self.bias_mat = nn.Parameter(torch.empty(dim * (dim + 1) // 2))
        self.feat_vec = nn.Parameter(torch.empty(dim))
        self.tril_emb = TrilEmbed(dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self.dim**-0.5
        nn.init.uniform_(self.bias_mat, -bound, bound)
        nn.init.uniform_(self.feat_vec, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1:] != (1,):
            raise ValueError(f"expected input shape (..., 1); got {tuple(x.shape)}")
        x = x[..., 0]

        a0 = self.tril_emb(self.bias_mat)
        p = F.softplus(self.feat_vec)

        mat = a0 + torch.diag_embed(x[..., None] * p)
        return torch.linalg.eigvalsh(mat)[..., self.eig_idx]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: ModelKind
    dim: int
    eig_idx: int | None = None

    @classmethod
    def from_kind_dim(cls, kind: ModelKind, dim: int):
        return cls(name=kind, kind=kind, dim=dim)


def make_model(spec: ModelSpec, input_dim: int) -> nn.Module:
    match spec.kind:
        case "unconstrained":
            return KthEigval(input_dim, spec.dim, spec.eig_idx)
        case "monotone":
            return KthEigval1DMonotone(spec.dim, spec.eig_idx)
        case _:
            raise ValueError(spec.kind)
