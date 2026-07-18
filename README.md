# spectral_neuron_paper

Research code for the spectral neuron paper.

## MovieLens scaling experiment

The MovieLens experiment is a matrix-completion study using only user and movie
identities to predict ratings. With exactly those two active fields, the FM is classical
biased matrix factorization:

```text
rating(user, movie) = global bias + user bias + movie bias
                    + dot(user embedding, movie embedding)
```

The fixed split is random 80/10/10 within each user. If a movie would otherwise be
absent from training, one of its holdout ratings is moved into training. Thus every
validation and test identity is estimable from the complete training pool, while each
data seed changes only the permutation of that pool. Nested prefixes make one pass over
that permutation. Prefix warm-coverage fractions are recorded because the complete
split is warm but an early random prefix need not be.

Users and movies are mapped to compact, disjoint ID ranges. There is no hashing or
fitted feature preprocessing. Ratings are shifted by the fixed midpoint of the official
0.5--5 scale for optimization; RMSE is unchanged and remains in rating units.
For a spectral dimension `d`, each identity has `d * (d + 1) // 2` matrix coordinates.
The matched FM rank is one less, because its per-identity linear bias consumes the
remaining parameter.

The runner accepts `ratings.csv`, its containing directory, or the official MovieLens
ZIP. The first invocation writes compact NumPy memory maps to a reusable cache.

```bash
uv run python -m paper.experiments.movielens_scaling \
  --data ~/datasets/ml-20m.zip \
  --profile sanity \
  --workers 2
```

The `small` profile is the inexpensive capacity pilot. The `full` profile runs from
roughly one million to 15.8 million training ratings; the recorded warm coverage shows
how much early checkpoints are still affected by unseen identities. Learning rates are
selected from final-checkpoint validation RMSE; selected configurations are then
initialized afresh and tested at every checkpoint. Use `--variant` with `linear`, `fm`,
or `spectral`, plus `--write-mode append`, to run the grid in shards.

## HIGGS scaling experiment

The HIGGS experiment consumes the headerless 11-million-row CSV. It uses the first
10 million rows for training, the next 500,000 for validation, and the official final
500,000-row test partition. This is the dataset's published row-order split, not a
chronological split.

The first run converts the CSV into float32 feature and uint8 label memory maps and
stores training-only means and standard deviations. The fixed standardizer is then
applied to every model and checkpoint. All 28 inputs remain numeric, including the
four ternary b-tag fields; there is no binning, one-hot encoding, or imputation. By
default this roughly 1.25 GB base cache, plus one 40 MB training-order file per data
seed, lives in
`.HIGGS.csv.cache-v1` beside the input. Use `--cache-dir` to place it elsewhere.

Each trajectory makes one pass over nested prefixes of a fixed shuffled training
order. The x-axis is therefore examples seen by the optimizer, not the number of rows
available to a separately fitted preprocessing-and-training pipeline. Learning rates
are selected by validation log loss at the largest checkpoint. The selected model is
then retrained once and tested at every checkpoint, so each plotted curve comes from
one coherent trajectory.

The comparison contains linear and spectral models plus one-, two-, and three-hidden-
layer ReLU MLP families. Hidden layers within an MLP have constant width. Widths are
computed from the requested spectral parameter budget and recorded with the actual
trainable parameter count in every result row.

```bash
uv run python -m paper.experiments.higgs_scaling \
  --data ~/datasets/HIGGS.csv \
  --profile sanity \
  --workers 2
```

The full profile is intentionally substantial. Run it in model-family shards and
append them to one explicitly named result file, for example:

```bash
uv run python -m paper.experiments.higgs_scaling \
  --data ~/datasets/HIGGS.csv \
  --profile full \
  --variant mlp-2 \
  --out notebooks/runs/higgs_scaling_full.csv \
  --write-mode append
```

Repeat the command for `linear`, `mlp-1`, `mlp-3`, and `spectral`. The
`notebooks/higgs_scaling.ipynb` companion validates the merged raw schema and run
completeness, derives its capacity table from recorded widths and parameter counts,
performs validation selection, and plots median test log loss or Brier score with
interquartile bands. This first version deliberately excludes leaderboard-oriented AUC
reporting and multi-epoch convergence studies.

## Criteo scaling experiment

The experiment expects the headerless, tab-separated training file from the Criteo
Display Advertising Challenge. The first run builds memory-mapped raw and encoded
caches beside the data; later trajectories reuse the encoded features directly. A full
five-seed, two-preprocessor cache occupies roughly 15 GB in addition to the raw cache.

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
values use standardized `log1p` magnitudes. Bucket features have implicit unit weights,
so their cached representation stores IDs only.

With progress enabled, the runner separately reports aggregate trajectory time spent in
training, validation, and test evaluation. These diagnostics are not written to the
result table, whose schema remains stable for appending and plotting.

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
