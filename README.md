# spectral_neuron_paper

Research code for the spectral neuron paper.

## Criteo scaling experiment

The experiment expects the headerless, tab-separated training file from the Criteo
Display Advertising Challenge. The first run builds a memory-mapped cache beside the
raw file; later runs reuse it.

Feature preprocessing is fitted once on a reproducible 10% sample of the chronological
training split. Each model then makes one pass over a fixed random permutation of that
split using Adam, with validation measurements taken at every requested training-size
checkpoint. Sparse embedding tables use SparseAdam, the sparse-gradient form of the same
optimizer.
Learning rates are selected on a small tuning seed grid and then frozen. The full profile
uses one eighth of the training partition (up to 4.58M impressions) and evaluates the
selected configurations on a crossed grid of four held-out data-order seeds and six
held-out initialization seeds.
The default run compares five variants: linear, FM, and spectral models with bucketed
numerics, plus linear and spectral models with hybrid numerical preprocessing. In the
hybrid representation, missing, zero, and negative values are indicators while positive
values use standardized `log1p` magnitudes.

```bash
uv run python -m paper.experiments.criteo_scaling \
  --data /path/to/train.txt \
  --profile sanity \
  --workers 2
```

Use `--cache-dir` to place the cache elsewhere, `--out` to override the raw CSV path,
and `--summary-out` to also write validation-selected aggregate results. Pass, for
example, `--variant spectral-new` to run one variant; its name is included in the
default output filename.
