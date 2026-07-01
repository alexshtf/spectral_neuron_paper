import itertools
import math

import fitstream as fts
import numpy as np
import pandas as pd
import torch
from torch import nn


class Synthetic1DStream:
    def __init__(
        self,
        func,
        lower=-4,
        upper=4,
        batch_size=32,
        noise_std=0.0,
        test_res=10000,
        rng=None,
    ):
        self.func = func
        self.lower = lower
        self.upper = upper
        self.batch_size = batch_size
        self.noise_std = noise_std
        if rng is None:
            self.rng = np.random.default_rng(42)
        else:
            self.rng = rng
        self.xs_test = torch.linspace(lower, upper, test_res).reshape(-1, 1)
        self.ys_test = torch.as_tensor(func(self.xs_test.numpy())).squeeze(-1)

    def train_batches(self):
        while True:
            xs_batch = self.rng.uniform(
                self.lower, self.upper, size=(self.batch_size, 1)
            )
            ys_batch = self.func(xs_batch)
            if self.noise_std > 0:
                ys_batch += self.rng.normal(0, self.noise_std, size=ys_batch.shape)
            yield torch.as_tensor(xs_batch), torch.as_tensor(ys_batch).squeeze(-1)

    def test(self, loss_fn, model):
        with torch.no_grad():
            ys_pred = model(self.xs_test)
            loss = loss_fn(ys_pred, self.ys_test)
        return loss.mean().item()


def train_on_stream(
    stream_provider,
    model,
    lr=1e-3,
):
    # setup variables for mini-batch loop
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # run training loop
    for step, (xs_batch, ys_batch) in enumerate(
        stream_provider.train_batches(), start=1
    ):
        loss = criterion(model(xs_batch), ys_batch)
        optim.zero_grad()
        loss.backward()
        optim.step()

        test_mse = stream_provider.test(criterion, model)

        yield {
            "step": step,
            "test_rmse": math.sqrt(test_mse.item()),
        }


def compute_scaling_law(model_fn, stream_provider, n_batches=200):
    train_logs = []
    for lr in np.geomspace(1e-4, 1e-1, 10):
        events = fts.pipe(
            train_on_stream(stream_provider, model_fn(), lr=lr), fts.take(n_batches)
        )
        train_log = fts.collect_pd(events)
        train_logs.append(train_log)

    return (
        pd.concat(train_logs, axis=0)
        .groupby("step")["test_rmse"]
        .min()
        .sort_index()
        .cummin()
        .to_frame()
    )
