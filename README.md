# spectral_neuron_paper

Research code for the spectral neuron paper.

## Real-data scaling protocol

The HIGGS and Criteo runners use the `repeated_shuffle` protocol.
Within one trajectory, `data_seed` initializes a deterministic stream of successive
fresh permutations of the fixed training pool. The permutations are concatenated and
batched as one stream rather than batched separately; a minibatch may therefore cross a
pass boundary. The first permutation preserves the earlier seeded one-pass order, while
later permutations are fresh and deterministic. The profile's explicit `train_sizes`
are the only evaluation checkpoints, and their maximum is the final training budget.
The runner automatically creates as many permutations as that budget requires; dataset
pass boundaries do not add checkpoints or change the endpoint. Nonterminal checkpoint
requests must lie on global minibatch boundaries; the final checkpoint may end with a
partial minibatch. Result rows report the explicit `train_size` counts together with the
fixed `train_pool_size`. Thus the x-axis is examples seen by the optimizer, not the
number of distinct available rows.

At every resolved `train_size`, each model-capacity configuration selects the learning
rate with the best median validation log loss across tuning seeds. Evaluation then
starts fresh trajectories and reports test metrics only at the checkpoints assigned to
their selected rate. Checkpoints that
select the same rate share one evaluation trajectory; when the selected rate changes,
the plotted curve stitches checkpoints from different trajectories. The result is a
validation-selected performance envelope over exact examples-seen budgets, not one
coherent optimization path or independently fitted dataset-size scaling.

When running model-family shards, point every command at the same explicit
repeated-shuffle output. Use `--write-mode overwrite` for the first shard and
`--write-mode append` only for later shards. Append mode checks that a nonempty CSV's
header exactly matches the new result columns. Repeated-shuffle rows include
`train_pool_size`, so do not append them to legacy one-pass results. Existing locally
generated repeated-shuffle files with injected pass-boundary checkpoints use the old
contract and must be overwritten before analysis. The committed unsuffixed CSVs under
`notebooks/runs/` are retained as historical provenance and are unsuitable for claims
that rely on the aligned protocol.

## HIGGS scaling experiment

The HIGGS experiment consumes the headerless 11-million-row CSV, either directly or
Zstandard-compressed with a `.zstd` suffix. Compressed input is decompressed as the CSV
stream is read; no expanded source file is written. It preserves the published
convention of reserving the final 500,000 rows for test. This repo defines its own
validation slice as the preceding 500,000 rows and uses the first 10 million for
training; the published convention does not define that validation boundary. This is a
row-order split, not a chronological split.

The first run converts the CSV into float32 feature and uint8 label memory maps and
stores training-only means and standard deviations. The fixed standardizer is then
applied to every model and checkpoint. All 28 inputs remain numeric, including the
four ternary b-tag fields; there is no binning, one-hot encoding, or imputation. By
default this roughly 1.25 GB base cache lives in `.HIGGS.csv.cache-v1` beside the input.
Use `--cache-dir` to place it elsewhere. Existing base caches remain valid; repeated
shuffling adds versioned order caches separately.

The shared per-checkpoint validation-selection contract applies. The full profile
evaluates its explicit power-of-two grid through `2**26` and stops there. Because that
budget exceeds the 10-million-row training pool, the stream continues through
successive deterministic shuffles.

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
  --variant linear \
  --out notebooks/runs/higgs_scaling_full_repeated_shuffle.csv \
  --write-mode overwrite
uv run python -m paper.experiments.higgs_scaling \
  --data ~/datasets/HIGGS.csv \
  --profile full \
  --variant mlp-1 \
  --out notebooks/runs/higgs_scaling_full_repeated_shuffle.csv \
  --write-mode append
```

Append `mlp-2`, `mlp-3`, and `spectral` in the same way. The
`notebooks/higgs_scaling.ipynb` companion validates the merged raw schema and run
completeness, derives its capacity table from recorded widths and parameter counts,
performs validation selection, and plots median test log loss or Brier score with
interquartile bands. The experiment deliberately focuses on log loss and Brier rather
than leaderboard-oriented AUC reporting.

### HIGGS feature-sensitivity bounds

The robustness runner loads a completed scaling profile, takes each spectral
dimension's validation-selected learning rate at the final checkpoint, and retrains
the profile's evaluation seeds to that budget using the same corpus and shuffle
caches. For the CLI-selected maximum magnitude `ε`, it draws one deterministic signed
perturbation `δ ~ Uniform(-ε, ε)` per standardized test row and feature, then compares
the logit change with the corresponding spectral-norm bound using the realized `|δ|`.

```bash
uv run python -m paper.experiments.higgs_robustness \
  --data ~/datasets/HIGGS.csv \
  --profile full \
  --noise-level 0.5 \
  --workers 4
```

The compressed result stores a `16 × 100` joint histogram over `|δ| ∈ [0, ε]` and
deviation ratio in `[0, 1]` for every seed, feature, and dimension, together with
zero-bound, above-bound, and maximum-ratio diagnostics.
`notebooks/higgs_robustness.ipynb` validates that file, reports the diagnostics, and
exports a publication PDF and PNG.
Its `shell_count` parameter merges adjacent magnitude bins into disjoint ranges; it
never interprets a shell as a cumulative upper bound. Any divisor of 16 is supported
(`1`, `2`, `4`, `8`, or `16`). Each feature-by-shell cell plots the raw histogram
probabilities and chooses its own y-axis limit; dimensions within a cell share that
limit. `feature_row_height_mm` controls the vertical space per feature and defaults to
12 mm.

## Criteo scaling experiment

The experiment expects the headerless, tab-separated training file from the Criteo
Display Advertising Challenge, either directly or Zstandard-compressed with a `.zstd`
suffix. Compressed input is decompressed as the TSV stream is read; no expanded source
file is written. The first run builds memory-mapped raw and encoded caches beside the
data; later trajectories reuse the encoded features directly. Fitted preprocessors are
stored as `.pkl.zstd` at Zstandard level 3 because they are loaded wholly into memory.
Legacy `.pkl` preprocessors are migrated by streaming them into the compressed format.
Memory-mapped data caches remain uncompressed.
The canonical train encoding for repeated shuffling is the seed- and pass-independent
`encoded-v4` cache. It stores field-local feature IDs as uint16 values and restores
their fixed global field offsets in each int32 training batch. Older encoded-cache
directories and hashed preprocessors may coexist with the exact-mapping caches and can
be removed later if their disk space is needed. The first exact-mapping run rebuilds
the fitted preprocessors and encoded features while reusing the raw corpus and shuffle
caches.

Feature preprocessing is fitted once on a reproducible 10% sample of the chronological
training split. Retained categorical values receive exact field-local IDs; missing
values and rare or unseen values use separate reserved IDs. Bucket numerics retain
their winner-style log-squared transformation, then use exact IDs over each field's
fitted transformed range, with separate missing and out-of-range IDs. Each model then
consumes the repeated-shuffle stream using Adam, with validation measurements taken at
every requested examples-seen checkpoint. Dense parameters use Adam and sparse
embedding tables use SparseAdam.
The full profile evaluates its explicit power-of-two grid through `2**28` and stops
there; the stream automatically creates however many training-pool permutations that
budget requires. Its four evaluation data-order seeds are held out from tuning.
Evaluation initialization seeds are 3 through 8: seeds 3 through 7 overlap tuning,
while only seed 8 is new.
The default run compares five variants: linear, FM, and spectral models with bucketed
numerics, plus linear and spectral models with hybrid numerical preprocessing. In the
hybrid representation, missing, zero, and negative values are indicators while positive
values use standardized `log1p` magnitudes. Bucket features have implicit unit weights,
so their cached representation stores IDs only.

With progress enabled, the runner separately reports aggregate trajectory time spent in
training, validation, and test evaluation. These diagnostics are not written to the
result table.

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

Use `--cache-dir` to place the cache elsewhere and `--out` to set the result CSV path.
For a sharded full run, start with `linear-bucketed`, then append
`linear-continuous` to the same output:

```bash
uv run python -m paper.experiments.criteo_scaling \
  --data /path/to/train.txt --profile full --variant linear-bucketed \
  --out notebooks/runs/criteo_scaling_full_repeated_shuffle.csv \
  --write-mode overwrite
uv run python -m paper.experiments.criteo_scaling \
  --data /path/to/train.txt --profile full --variant linear-continuous \
  --out notebooks/runs/criteo_scaling_full_repeated_shuffle.csv \
  --write-mode append
```

Append `fm`, `spectral-bucketed`, and `spectral-continuous` in the same way. The
notebook performs validation selection and aggregation directly from this raw result
table.
