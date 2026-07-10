from dataclasses import dataclass
from typing import Literal

import torch
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


def square_plus(x: torch.Tensor) -> torch.Tensor:
    return (torch.hypot(torch.as_tensor(1.), x) + x) / 2


class KthEigvalLastMonotone(nn.Module):
    """
    f(x) = lambda_k(A0 + x1 A1 + ... + xn An), An = diag(p), p >= 0.

    The last input feature is monotone. eig_idx is zero-based.
    """

    def __init__(
        self, num_features: int, dim: int, eig_idx: int | None = None
    ) -> None:
        super().__init__()
        if num_features < 1:
            raise ValueError(f"num_features must be positive; got {num_features}")

        self.num_features = num_features
        self.dim = dim
        self.eig_idx = dim // 2 if eig_idx is None else eig_idx
        if not 0 <= self.eig_idx < dim:
            raise ValueError(f"eig_idx must be in [0, {dim}); got {self.eig_idx}")

        num_tril = dim * (dim + 1) // 2
        self.dense_tril = nn.Parameter(torch.empty(num_features, num_tril))
        self.last_diag = nn.Parameter(torch.empty(dim))
        self.tril_emb = TrilEmbed(dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self.dim**-0.5
        nn.init.uniform_(self.dense_tril, -bound, bound)
        nn.init.uniform_(self.last_diag, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1:] != (self.num_features,):
            raise ValueError(
                f"expected input shape (..., {self.num_features}); got {tuple(x.shape)}"
            )

        # compute batchd A_0 + x_11 A_1 + ... + x_{n-1} A_{n-1}
        all_but_last_tril = self.dense_tril[0] + x[..., :-1].matmul(self.dense_tril[1:])
        all_but_last_mat = self.tril_emb(all_but_last_tril)

        # compute x_n diag(a_n)
        last_mat = torch.diag_embed(x[..., -1:] * square_plus(self.last_diag))

        # compute eigenvalues
        return torch.linalg.eigvalsh(all_but_last_mat + last_mat)[..., self.eig_idx]


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
            return KthEigvalLastMonotone(input_dim, spec.dim, spec.eig_idx)
        case _:
            raise ValueError(spec.kind)
