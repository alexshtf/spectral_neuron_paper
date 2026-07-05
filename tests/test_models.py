import pytest
import torch

from paper.models import KthEigval1DMonotone, TrilEmbed


def test_tril_embed_outputs_symmetric():
    embed = TrilEmbed(4)
    x = torch.randn(3, 4 * 5 // 2)

    mat = embed(x)

    assert mat.shape == (3, 4, 4)
    assert torch.allclose(mat, mat.transpose(-1, -2))


def test_monotone_1d_model_is_monotone_on_grid():
    torch.manual_seed(0)
    model = KthEigval1DMonotone(dim=5)
    x = torch.linspace(-4.0, 4.0, 200).reshape(-1, 1)

    with torch.no_grad():
        y = model(x)

    assert torch.all(y[1:] >= y[:-1] - 1e-5)


def test_monotone_1d_model_preserves_leading_shape():
    torch.manual_seed(0)
    model = KthEigval1DMonotone(dim=5)

    with torch.no_grad():
        assert model(torch.tensor([0.0])).shape == ()
        assert model(torch.zeros(7, 1)).shape == (7,)
        assert model(torch.zeros(2, 3, 1)).shape == (2, 3)


def test_monotone_1d_model_rejects_ambiguous_vector_input():
    model = KthEigval1DMonotone(dim=5)

    with pytest.raises(ValueError, match=r"expected input shape \(\.\.\., 1\)"):
        model(torch.zeros(7))
