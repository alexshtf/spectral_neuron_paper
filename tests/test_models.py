import pytest
import torch

from paper.models import (
    FactorizationMachine,
    KthEigval,
    KthEigvalLastMonotone,
    SparseLinear,
    SparseMiddleEigval,
    TrilEmbed,
    make_seeded_model,
    matched_fm_rank,
    square_plus,
)


def test_make_seeded_model_is_reproducible_without_changing_the_ambient_rng():
    torch.manual_seed(17)
    state = torch.random.get_rng_state().clone()

    first = make_seeded_model(lambda: torch.nn.Linear(3, 2), seed=3)
    second = make_seeded_model(lambda: torch.nn.Linear(3, 2), seed=3)

    assert torch.equal(torch.random.get_rng_state(), state)
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter)


def test_tril_embed_is_isometric():
    embed = TrilEmbed(4)
    x = torch.randn(3, 4 * 5 // 2)

    mat = embed(x)

    assert mat.shape == (3, 4, 4)
    assert torch.allclose(mat, mat.transpose(-1, -2))
    torch.testing.assert_close(
        mat.square().sum(dim=(-2, -1)), x.square().sum(dim=-1)
    )


def _assert_scalar_plus_jitter(
    diagonals: torch.Tensor,
    *,
    coefficient_lower: float,
    coefficient_upper: float,
    jitter_bound: float,
) -> None:
    """Check that each row admits diagonal = alpha + epsilon."""
    feasible_lower = torch.clamp_min(
        diagonals.amax(dim=-1) - jitter_bound,
        coefficient_lower,
    )
    feasible_upper = torch.clamp_max(
        diagonals.amin(dim=-1) + jitter_bound,
        coefficient_upper,
    )
    assert torch.all(feasible_lower <= feasible_upper)
    assert torch.all(diagonals.amax(dim=-1) > diagonals.amin(dim=-1))


def _assert_centered_gapped_jittered_initialization(
    base_tril: torch.Tensor,
    feature_tril: torch.Tensor,
    *,
    dim: int,
    eig_idx: int,
    fan_in: int,
) -> torch.Tensor:
    embed = TrilEmbed(dim)
    base = embed(base_tril)
    features = embed(feature_tril)

    expected_spectrum = torch.arange(dim).sub(eig_idx).sign().to(base)
    torch.testing.assert_close(torch.linalg.eigvalsh(base), expected_spectrum)

    diagonals = features.diagonal(dim1=-2, dim2=-1)
    torch.testing.assert_close(features, torch.diag_embed(diagonals))
    coefficient_bound = fan_in**-0.5
    _assert_scalar_plus_jitter(
        diagonals,
        coefficient_lower=-coefficient_bound,
        coefficient_upper=coefficient_bound,
        jitter_bound=1 / (20 * fan_in),
    )

    commutators = base @ features - features @ base
    assert torch.all(torch.linalg.matrix_norm(commutators) > 1e-6)
    return base


@pytest.mark.parametrize("eig_idx", [0, 2, 4])
def test_dense_spectral_initialization_preserves_gap_and_breaks_commutation(
    eig_idx,
):
    torch.manual_seed(0)
    model = KthEigval(num_features=3, dim=5, eig_idx=eig_idx)
    _assert_centered_gapped_jittered_initialization(
        model.lin.bias,
        model.lin.weight.mT,
        dim=model.dim,
        eig_idx=model.eig_idx,
        fan_in=model.lin.in_features,
    )

    bounds = torch.tensor([-5.0, 5.0])
    corners = torch.cartesian_prod(
        *(bounds for _ in range(model.lin.in_features))
    )
    eigvals = torch.linalg.eigvalsh(model.tril_emb(model.lin(corners)))
    selected = eigvals[..., eig_idx]
    gaps = []
    if eig_idx > 0:
        gaps.append(selected - eigvals[..., eig_idx - 1])
    if eig_idx + 1 < model.dim:
        gaps.append(eigvals[..., eig_idx + 1] - selected)
    assert torch.stack(gaps).amin() >= 0.5 - 1e-6


def test_sparse_spectral_initialization_uses_active_field_jitter_scale():
    torch.manual_seed(0)
    model = SparseMiddleEigval(
        num_features=7,
        num_fields=2,
        dim=5,
    )
    _assert_centered_gapped_jittered_initialization(
        model.base_tril,
        model.feature_tril.weight,
        dim=model.dim,
        eig_idx=model.dim // 2,
        fan_in=model.num_fields,
    )


def test_monotone_spectral_initialization_jitters_positive_diagonal():
    torch.manual_seed(0)
    model = KthEigvalLastMonotone(num_features=3, dim=5, eig_idx=2)
    base = _assert_centered_gapped_jittered_initialization(
        model.base_tril,
        model.feature_tril,
        dim=model.dim,
        eig_idx=model.eig_idx,
        fan_in=model.num_features,
    )

    monotone_diagonal = square_plus(model.last_diag)
    coefficient_bound = model.num_features**-0.5
    _assert_scalar_plus_jitter(
        monotone_diagonal.unsqueeze(0),
        coefficient_lower=coefficient_bound / 2,
        coefficient_upper=coefficient_bound,
        jitter_bound=1 / (20 * model.num_features),
    )
    assert torch.all(monotone_diagonal > 0)

    monotone_matrix = torch.diag(monotone_diagonal)
    commutator = base @ monotone_matrix - monotone_matrix @ base
    assert torch.linalg.matrix_norm(commutator) > 1e-6


def test_last_monotone_model_matches_matrix_path():
    model = KthEigvalLastMonotone(num_features=3, dim=2, eig_idx=1)
    sqrt_2 = 2**0.5
    with torch.no_grad():
        model.base_tril.copy_(torch.tensor([1.0, sqrt_2 * 0.2, 2.0]))
        model.feature_tril.copy_(
            torch.tensor(
                [
                    [0.5, sqrt_2 * -0.4, -0.25],
                    [-1.0, sqrt_2 * 0.3, 0.75],
                ]
            )
        )
        model.last_diag.copy_(torch.tensor([0.0, 0.75]))

    x = torch.tensor([2.0, -1.0, 3.0])
    a0 = torch.tensor([[1.0, 0.2], [0.2, 2.0]])
    a1 = torch.tensor([[0.5, -0.4], [-0.4, -0.25]])
    a2 = torch.tensor([[-1.0, 0.3], [0.3, 0.75]])
    a3 = torch.diag(torch.tensor([0.5, 1.0]))
    expected = torch.linalg.eigvalsh(a0 + x[0] * a1 + x[1] * a2 + x[2] * a3)[1]

    assert torch.allclose(model(x), expected)


@pytest.mark.parametrize("num_features", [1, 3])
def test_last_monotone_model_is_monotone_in_last_feature(num_features):
    torch.manual_seed(0)
    model = KthEigvalLastMonotone(num_features=num_features, dim=5)
    x = torch.zeros(200, num_features)
    x[..., :-1] = torch.randn(num_features - 1)
    x[..., -1] = torch.linspace(-4.0, 4.0, 200)

    with torch.no_grad():
        y = model(x)

    assert torch.all(y[1:] >= y[:-1] - 1e-5)


def test_last_monotone_model_preserves_leading_shape():
    torch.manual_seed(0)
    model = KthEigvalLastMonotone(num_features=3, dim=5)

    with torch.no_grad():
        assert model(torch.zeros(3)).shape == ()
        assert model(torch.zeros(7, 3)).shape == (7,)
        assert model(torch.zeros(2, 3, 3)).shape == (2, 3)


def test_last_monotone_model_rejects_wrong_feature_count():
    model = KthEigvalLastMonotone(num_features=3, dim=5)

    with pytest.raises(ValueError, match=r"expected input shape \(\.\.\., 3\)"):
        model(torch.zeros(7, 2))


def test_sparse_models_preserve_leading_shape():
    feature_ids = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
    feature_values = torch.ones_like(feature_ids, dtype=torch.float32)

    assert SparseLinear(5, 2)(feature_ids, feature_values).shape == (2, 2)
    assert (
        FactorizationMachine(5, 2, rank=3)(feature_ids, feature_values).shape
        == (2, 2)
    )
    assert SparseMiddleEigval(5, 2, dim=3)(feature_ids, feature_values).shape == (
        2,
        2,
    )


def test_sparse_models_treat_implicit_values_as_unit_weights():
    feature_ids = torch.tensor([[0, 1, 2]])
    unit_weights = torch.ones_like(feature_ids, dtype=torch.float32)

    for model in (
        SparseLinear(3, 3),
        FactorizationMachine(3, 3, rank=2),
        SparseMiddleEigval(3, 3, dim=3),
    ):
        assert torch.allclose(
            model(feature_ids), model(feature_ids, unit_weights)
        )


@pytest.mark.parametrize("dim", [3, 5, 9, 15])
def test_fm_and_spectral_match_parameters_per_feature(dim):
    num_features = 17
    parameters_per_feature = dim * (dim + 1) // 2
    fm = FactorizationMachine(
        num_features,
        num_fields=3,
        rank=matched_fm_rank(dim),
    )
    spectral = SparseMiddleEigval(num_features, num_fields=3, dim=dim)

    fm_lookup_parameters = fm.weight.weight.numel() + fm.embedding.weight.numel()
    spectral_lookup_parameters = spectral.feature_tril.weight.numel()

    assert fm_lookup_parameters == spectral_lookup_parameters
    assert spectral_lookup_parameters == parameters_per_feature * num_features


def test_factorization_machine_matches_pairwise_definition():
    model = FactorizationMachine(num_features=4, num_fields=3, rank=2)
    with torch.no_grad():
        model.bias.fill_(0.25)
        model.weight.weight.copy_(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
        model.embedding.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0], [0.0, 0.0]])
        )

    ids = torch.tensor([[0, 1, 2]])
    values = torch.tensor([[1.0, 2.0, 0.5]])
    vectors = model.embedding(ids[0]) * values[0, :, None]
    expected_interaction = sum(
        torch.dot(vectors[i], vectors[j])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    expected = 0.25 + 1.0 + 2.0 * 2.0 + 0.5 * 3.0 + expected_interaction

    assert torch.allclose(model(ids, values), expected.reshape(1))


def test_sparse_spectral_model_sums_feature_matrices():
    model = SparseMiddleEigval(num_features=3, num_fields=2, dim=2)
    sqrt_2 = 2**0.5
    with torch.no_grad():
        model.base_tril.copy_(torch.tensor([1.0, 0.0, 2.0]))
        model.feature_tril.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, sqrt_2, 0.0],
                    [0.0, sqrt_2, 3.0],
                ]
            )
        )

    ids = torch.tensor([[0, 2]])
    values = torch.tensor([[2.0, 0.5]])
    expected = torch.linalg.eigvalsh(torch.tensor([[3.0, 0.5], [0.5, 3.5]]))[1]

    assert torch.allclose(model(ids, values), expected.reshape(1))
