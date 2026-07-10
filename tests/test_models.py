import pytest
import torch

from paper.models import KthEigvalLastMonotone, TrilEmbed


def test_tril_embed_outputs_symmetric():
    embed = TrilEmbed(4)
    x = torch.randn(3, 4 * 5 // 2)

    mat = embed(x)

    assert mat.shape == (3, 4, 4)
    assert torch.allclose(mat, mat.transpose(-1, -2))


def test_last_monotone_model_matches_matrix_path():
    model = KthEigvalLastMonotone(num_features=3, dim=2, eig_idx=1)
    with torch.no_grad():
        model.dense_tril.copy_(
            torch.tensor(
                [
                    [1.0, 0.2, 2.0],
                    [0.5, -0.4, -0.25],
                    [-1.0, 0.3, 0.75],
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
