import torch
from functional_geometry.fisher import categorical_fisher_product, fisher_from_score_gradients, fisher_matrix_vector


def test_categorical_fisher_is_symmetric_psd():
    p = torch.tensor([0.1, 0.2, 0.7])
    u, v = torch.tensor([1.0, -2.0, 0.5]), torch.tensor([-0.2, 0.3, 1.1])
    assert torch.allclose(u @ categorical_fisher_product(p, v), v @ categorical_fisher_product(p, u), atol=1e-6)
    assert v @ categorical_fisher_product(p, v) >= 0


def test_score_gradient_fisher_is_symmetric_psd_and_deterministic():
    generator = torch.Generator().manual_seed(7)
    gradients = torch.randn(64, 8, generator=generator)
    first = fisher_from_score_gradients(gradients)
    second = fisher_from_score_gradients(gradients)
    u, v = torch.randn(8, generator=generator), torch.randn(8, generator=generator)
    assert torch.equal(first.eigenvalues, second.eigenvalues)
    assert torch.allclose(u @ fisher_matrix_vector(first, v), v @ fisher_matrix_vector(first, u), atol=1e-5)
    assert v @ fisher_matrix_vector(first, v) >= -1e-6

