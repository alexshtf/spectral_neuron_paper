# spectral_neuron_paper

Research code for the spectral neuron paper.

## Criteo scaling experiment

The experiment expects the headerless, tab-separated training file from the Criteo
Display Advertising Challenge. The first run builds a memory-mapped cache beside the
raw file; later runs reuse it.

Feature preprocessing is fitted once on a reproducible 10% sample of the chronological
training split. Each model then makes one pass over a fixed random permutation of that
split using Adam, with validation measurements taken at every requested training-size
checkpoint. Dense parameters use Adam and sparse embedding tables use SparseAdam.
Learning rates are selected on a small tuning seed grid and then frozen. The full profile
uses one eighth of the training partition (up to 4.58M impressions) and evaluates the
selected configurations on a crossed grid of four held-out data-order seeds and six
held-out initialization seeds.
The default run compares five variants: linear, FM, and spectral models with bucketed
numerics, plus linear and spectral models with hybrid numerical preprocessing. In the
hybrid representation, missing, zero, and negative values are indicators while positive
values use standardized `log1p` magnitudes.

Result rows use one `dim` column for the matched nonlinear capacity: it is the spectral
matrix dimension, and the corresponding FM rank is `dim * (dim + 1) // 2 - 1` (`0` for
the dimensionless linear baselines). Other capacity and preprocessing labels are derived
from `model` and `dim` rather than duplicated in the table.

```bash
uv run python -m paper.experiments.criteo_scaling \
  --data /path/to/train.txt \
  --profile sanity \
  --workers 2
```

Use `--cache-dir` to place the cache elsewhere and `--out` to override the raw CSV path.
Pass, for example, `--variant spectral-new` to run one variant; its name is included in
the default output filename. Use `--write-mode append` to extend an existing compatible
result file instead of overwriting it. The notebook performs validation selection and
aggregation directly from this raw result table.
