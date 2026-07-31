from dataclasses import dataclass
from math import sqrt
from typing import Literal

import torch
from torch import nn

type ModelKind = Literal["unconstrained", "monotone"]


def _scale_tril_off_diagonal(
    x: torch.Tensor, diagonal: torch.Tensor, factor: float
) -> torch.Tensor:
    return torch.where(diagonal, x, x * factor)


def _resolve_eig_idx(dim: int, eig_idx: int | None) -> int:
    eig_idx = dim // 2 if eig_idx is None else eig_idx
    if not 0 <= eig_idx < dim:
        raise ValueError(f"eig_idx must be in [0, {dim}); got {eig_idx}")
    return eig_idx


def _identity_bound(fan_in: int) -> float:
    return fan_in**-0.5


@torch.no_grad()
def _identity_coefficients(
    reference: torch.Tensor,
    shape: torch.Size | tuple[int, ...],
    *,
    fan_in: int,
) -> torch.Tensor:
    """Draw centered, fan-in-scaled coefficients for identity matrices."""
    bound = _identity_bound(fan_in)
    return reference.new_empty(shape).uniform_(-bound, bound)


@torch.no_grad()
def _squareplus_identity_coefficient(
    reference: torch.Tensor, *, fan_in: int
) -> torch.Tensor:
    """Draw a positive coefficient that is safe to invert through squareplus.

    The lower cutoff keeps the inverse c - 1 / (4c) away from its singularity
    at zero while keeping c within a factor of two of the fan-in scale.
    """
    bound = _identity_bound(fan_in)
    return reference.new_empty(()).uniform_(bound / 2, bound)


@torch.no_grad()
def _reset_spectral_pencil_(
    base_tril: torch.Tensor,
    feature_tril: torch.Tensor,
    *,
    dim: int,
    eig_idx: int,
    fan_in: int,
) -> None:
    """Set A0 = Q diag(sign(j - k)) Q.T and Ai = alpha_i I."""
    tril_i, tril_j = torch.tril_indices(dim, dim, device=base_tril.device)
    diag = tril_i == tril_j

    feature_tril.zero_()
    coefficients = _identity_coefficients(
        feature_tril,
        feature_tril.shape[:-1],
        fan_in=fan_in,
    )
    feature_tril[..., diag] = coefficients.unsqueeze(-1)

    q, r = torch.linalg.qr(base_tril.new_empty(dim, dim).normal_())
    signs = torch.where(r.diagonal() < 0, -1, 1)
    q = q * signs
    spectrum = torch.arange(dim, device=base_tril.device) - eig_idx
    spectrum = spectrum.sign().to(base_tril.dtype)
    base = (q * spectrum) @ q.mT
    base_tril.copy_(
        _scale_tril_off_diagonal(base[tril_i, tril_j], diag, sqrt(2))
    )


@torch.no_grad()
def _reset_positive_identity_(raw_diag: torch.Tensor, *, fan_in: int) -> None:
    coefficient = _squareplus_identity_coefficient(raw_diag, fan_in=fan_in)
    raw_diag.fill_(coefficient - 1 / (4 * coefficient))


class TrilEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        i, j = torch.tril_indices(dim, dim)

        grid = torch.zeros(dim, dim, dtype=torch.long)
        grid[i, j] = torch.arange(len(i))
        grid = torch.maximum(grid, grid.T)

        self.register_buffer("map", grid.flatten())
        self.register_buffer("diagonal", i == j, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Isometrically embed lower-triangular coordinates as symmetric matrices."""
        x = _scale_tril_off_diagonal(x, self.diagonal, 1 / sqrt(2))
        return x[..., self.map].view(*x.shape[:-1], self.dim, self.dim)


class KthEigval(nn.Module):
    def __init__(self, num_features: int, dim: int, eig_idx: int | None = None):
        super().__init__()
        self.dim = dim
        self.eig_idx = _resolve_eig_idx(dim, eig_idx)

        self.lin = nn.Linear(num_features, dim * (dim + 1) // 2)
        self.tril_emb = TrilEmbed(dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_spectral_pencil_(
            self.lin.bias,
            self.lin.weight.mT,
            dim=self.dim,
            eig_idx=self.eig_idx,
            fan_in=self.lin.in_features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mat = self.tril_emb(self.lin(x))
        eigvals = torch.linalg.eigvalsh(mat)
        return eigvals[..., self.eig_idx]


def _check_sparse_input(
    feature_ids: torch.Tensor,
    feature_values: torch.Tensor | None,
    num_fields: int,
) -> None:
    if feature_ids.shape[-1:] != (num_fields,):
        raise ValueError(
            f"expected input shape (..., {num_fields}); got {tuple(feature_ids.shape)}"
        )
    if feature_values is not None and feature_values.shape != feature_ids.shape:
        raise ValueError(
            f"feature values must have shape {tuple(feature_ids.shape)}; "
            f"got {tuple(feature_values.shape)}"
        )


def _weighted_lookup(
    table: nn.Embedding,
    feature_ids: torch.Tensor,
    feature_values: torch.Tensor | None,
) -> torch.Tensor:
    lookup = table(feature_ids)
    return (
        lookup
        if feature_values is None
        else lookup * feature_values.unsqueeze(-1)
    )


class SparseLinear(nn.Module):
    """A linear logit over weighted sparse features."""

    def __init__(self, num_features: int, num_fields: int):
        super().__init__()
        self.num_fields = num_fields
        self.weight = nn.Embedding(num_features, 1, sparse=True)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.weight.weight)

    def forward(
        self,
        feature_ids: torch.Tensor,
        feature_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_sparse_input(feature_ids, feature_values, self.num_fields)
        weighted = _weighted_lookup(
            self.weight, feature_ids, feature_values
        ).squeeze(-1)
        return weighted.sum(dim=-1) + self.bias


class FactorizationMachine(SparseLinear):
    """A second-order factorization machine with linear and latent weights."""

    def __init__(self, num_features: int, num_fields: int, rank: int):
        if rank < 1:
            raise ValueError(f"rank must be positive; got {rank}")
        super().__init__(num_features, num_fields)
        self.rank = rank
        self.embedding = nn.Embedding(num_features, rank, sparse=True)
        nn.init.normal_(self.embedding.weight, std=1e-2)

    def forward(
        self,
        feature_ids: torch.Tensor,
        feature_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        linear = super().forward(feature_ids, feature_values)
        vectors = _weighted_lookup(self.embedding, feature_ids, feature_values)
        interaction = 0.5 * (
            vectors.sum(dim=-2).square().sum(dim=-1)
            - vectors.square().sum(dim=(-2, -1))
        )
        return linear + interaction


class SparseKthEigval(nn.Module):
    """A spectral neuron whose active categorical features contribute matrices."""

    def __init__(
        self,
        num_features: int,
        num_fields: int,
        dim: int,
        eig_idx: int | None = None,
    ):
        super().__init__()
        self.num_fields = num_fields
        self.dim = dim
        self.eig_idx = _resolve_eig_idx(dim, eig_idx)

        num_tril = dim * (dim + 1) // 2
        self.feature_tril = nn.Embedding(num_features, num_tril, sparse=True)
        self.base_tril = nn.Parameter(torch.empty(num_tril))
        self.tril_emb = TrilEmbed(dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_spectral_pencil_(
            self.base_tril,
            self.feature_tril.weight,
            dim=self.dim,
            eig_idx=self.eig_idx,
            fan_in=self.num_fields,
        )

    def forward(
        self,
        feature_ids: torch.Tensor,
        feature_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_sparse_input(feature_ids, feature_values, self.num_fields)
        weighted = _weighted_lookup(
            self.feature_tril, feature_ids, feature_values
        )
        tril = self.base_tril + weighted.sum(dim=-2)
        return torch.linalg.eigvalsh(self.tril_emb(tril))[..., self.eig_idx]


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
        self.eig_idx = _resolve_eig_idx(dim, eig_idx)

        num_tril = dim * (dim + 1) // 2
        self.dense_tril = nn.Parameter(torch.empty(num_features, num_tril))
        self.last_diag = nn.Parameter(torch.empty(dim))
        self.tril_emb = TrilEmbed(dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_spectral_pencil_(
            self.dense_tril[0],
            self.dense_tril[1:],
            dim=self.dim,
            eig_idx=self.eig_idx,
            fan_in=self.num_features,
        )
        _reset_positive_identity_(self.last_diag, fan_in=self.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1:] != (self.num_features,):
            raise ValueError(
                f"expected input shape (..., {self.num_features}); got {tuple(x.shape)}"
            )

        all_but_last_tril = self.dense_tril[0] + x[..., :-1].matmul(self.dense_tril[1:])
        all_but_last_mat = self.tril_emb(all_but_last_tril)
        last_mat = torch.diag_embed(x[..., -1:] * square_plus(self.last_diag))
        return torch.linalg.eigvalsh(all_but_last_mat + last_mat)[..., self.eig_idx]


@dataclass(frozen=True)
class ModelSpec:
    kind: ModelKind
    dim: int
    eig_idx: int | None = None


def make_model(spec: ModelSpec, input_dim: int) -> nn.Module:
    match spec.kind:
        case "unconstrained":
            return KthEigval(input_dim, spec.dim, spec.eig_idx)
        case "monotone":
            return KthEigvalLastMonotone(input_dim, spec.dim, spec.eig_idx)
        case _:
            raise ValueError(spec.kind)
