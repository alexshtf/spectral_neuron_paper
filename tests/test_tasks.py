import numpy as np
import torch

from paper.tasks import make_univariate_task


def test_training_inputs_match_across_noise_levels():
    def make_task(noise_std):
        return make_univariate_task(
            lambda x: x[..., 0],
            lower=-1.0,
            upper=1.0,
            batch_size=4,
            val_size=3,
            test_size=5,
            seed=0,
            noise_std=noise_std,
        )

    noiseless = make_task(0.0).train_batches(np.random.default_rng(1))
    noisy = make_task(0.1).train_batches(np.random.default_rng(1))

    for _ in range(3):
        x_noiseless, _ = next(noiseless)
        x_noisy, _ = next(noisy)
        torch.testing.assert_close(x_noiseless, x_noisy)
