# Spectral neuron experiments

This repository contains the experiments for the spectral neuron paper (https://arxiv.org/abs/2608.08003): scalar models of
the form

```text
x ↦ λₖ(A₀ + x₁A₁ + ⋯ + xₙAₙ),
```

where the learned matrices are symmetric and `λₖ` selects one ordered
eigenvalue. The experiments study whether these models are straightforward to
train and improve with scale while retaining useful spectral structure, such as
shape control and explicit feature-sensitivity bounds. They are not intended as
state-of-the-art benchmark submissions.

## Setup

The project requires Python 3.14 and uses [uv](https://docs.astral.sh/uv/) for a
locked environment. From the repository root:

```bash
uv sync --locked
uv run pytest
```

Launch the notebooks from the repository root so their relative result paths
resolve correctly:

```bash
uv run jupyter lab
```

## Repository layout

- `src/paper/models.py` implements the spectral and comparison models.
- `src/paper/targets.py`, `tasks.py`, `training.py`, and `tuning.py` contain the
  reusable synthetic-data and training components.
- `src/paper/experiments/` contains the executable experiment modules.
- `notebooks/` loads saved results, summarizes them, and produces the figures.
- `notebooks/runs/` contains the retained paper results and their
  [provenance manifest](notebooks/runs/README.md).
- `plots/` and `visuals/` contain manuscript-ready figure exports.
- `tests/` protects the model mathematics and experiment-selection contracts.

## Synthetic experiments

The univariate and bivariate experiments compare unconstrained and
shape-constrained spectral models on randomly generated target functions. The
test inputs are fixed grids and the test targets are noiseless.

Quick checks:

```bash
mkdir -p local-runs
uv run python -m paper.experiments.univariate --profile sanity \
  --out local-runs/univariate_sanity.csv.zst
uv run python -m paper.experiments.bivariate --profile sanity \
  --out local-runs/bivariate_sanity.csv.zst
```

Paper profiles:

```bash
uv run python -m paper.experiments.univariate --profile full \
  --out local-runs/univariate_full.csv.zst
uv run python -m paper.experiments.bivariate --profile full \
  --out local-runs/bivariate_full.csv.zst
```

Pandas reads these Zstandard-compressed `.csv.zst` files directly. Explicit
local output paths keep reruns separate from the frozen paper results.

## Real-data experiments

Set the two source paths before using the commands below:

```bash
HIGGS=/path/to/higgs.csv.zstd
CRITEO=/path/to/train.txt.zstd
```

Both inputs must be headerless and may be either uncompressed or compressed
with the `.zstd` suffix. The first run streams each source into memory-mapped
arrays and stores its cache beside the source. Pass `--cache-dir` to put the
cache elsewhere.

Download [`HIGGS.csv.gz` from UCI](https://archive.ics.uci.edu/dataset/280/higgs),
then decompress it to `HIGGS.csv` or recompress the CSV stream as `.zstd`.
Download `kaggle-display-advertising-challenge-dataset.tar.gz` from
[Criteo's dataset page](https://ailab.criteo.com/ressources/) under its terms,
extract the headerless TSV `train.txt`, and use it directly or recompress it as
`.zstd`. The similarly named Criteo 1-TB dataset is different.

HIGGS uses the first 10 million rows for training, the next 500,000 for
validation, and the final 500,000 for test. All 28 fields remain numeric and
are standardized using training-only statistics.

Criteo uses the first 80% of rows for training, the next 10% for validation,
and the final 10% for test. Its 13 numeric and 26 categorical fields are fitted
on a reproducible sample of the training split. Categorical values receive
exact field-local IDs; numeric fields use either buckets or a hybrid continuous
representation.

Quick checks:

```bash
uv run python -m paper.experiments.higgs_scaling \
  --data "$HIGGS" --profile sanity \
  --out local-runs/higgs_scaling_sanity.csv

uv run python -m paper.experiments.criteo_scaling \
  --data "$CRITEO" --profile sanity \
  --out local-runs/criteo_scaling_sanity.csv
```

The full scaling experiments can be run in model-family shards. Start each
result with `overwrite`, then append the remaining families:

```bash
HIGGS_RESULTS=local-runs/higgs_scaling_full_repeated_shuffle.csv

uv run python -m paper.experiments.higgs_scaling \
  --data "$HIGGS" --profile full --variant linear \
  --out "$HIGGS_RESULTS" --write-mode overwrite

for variant in mlp-1 mlp-2 mlp-3 spectral; do
  uv run python -m paper.experiments.higgs_scaling \
    --data "$HIGGS" --profile full --variant "$variant" \
    --out "$HIGGS_RESULTS" --write-mode append
done
```

```bash
CRITEO_RESULTS=local-runs/criteo_scaling_full_repeated_shuffle.csv

uv run python -m paper.experiments.criteo_scaling \
  --data "$CRITEO" --profile full --variant linear-bucketed \
  --out "$CRITEO_RESULTS" --write-mode overwrite

for variant in linear-continuous fm spectral-bucketed spectral-continuous; do
  uv run python -m paper.experiments.criteo_scaling \
    --data "$CRITEO" --profile full --variant "$variant" \
    --out "$CRITEO_RESULTS" --write-mode append
done
```

After the full HIGGS scaling run, reproduce the feature-sensitivity experiment:

```bash
HIGGS_RESULTS=local-runs/higgs_scaling_full_repeated_shuffle.csv

uv run python -m paper.experiments.higgs_robustness \
  --data "$HIGGS" \
  --profile full \
  --scaling-results "$HIGGS_RESULTS" \
  --noise-level 0.5 \
  --out local-runs/higgs_robustness_full_noise_0p5_repeated_shuffle.csv.zst
```

This measures the observed change in each test logit against its spectral-norm
feature bound over the complete 500,000-row test split. The default result is a
compressed `.csv.zst` histogram artifact under `notebooks/runs/`.

## Scaling protocol

HIGGS and Criteo use the `repeated_shuffle` protocol. A `data_seed` defines a
deterministic sequence of fresh permutations of the fixed training pool. These
permutations form one continuous minibatch stream, so `train_size` means the
exact number of examples seen by the optimizer rather than the number of
distinct rows.

At each requested `train_size`, each model and capacity selects the learning
rate with the lowest median validation loss across tuning seeds. Fresh
evaluation trajectories then report test metrics only for the selected rate.
Learning rates, checkpoints, and summaries are therefore selected without test
data. Reported curves aggregate evaluation seeds with medians and interquartile
ranges.

The full profiles are substantial. `--workers` runs independent trajectories
in parallel, while `--quiet` suppresses progress output. Each module exposes
all available options through `--help`.

## Results and notebooks

The committed result manifest records each retained artifact, profile,
producing commit, row count, and source dataset identity. Generated local runs
are not evidence for the paper unless they satisfy the same profile and are
recorded there. For byte-for-byte reproduction of a frozen artifact, check out
its listed producing commit; later implementation improvements can change rerun
values without changing the experiment protocol.

The notebooks are deliberately analysis-only: they read the retained results,
validate the expected experiment grid, summarize across seeds, and plot. To
execute and save one from the repository root, run for example:

```bash
uv run jupyter execute --inplace notebooks/univariate_fitting.ipynb
```

The retained notebooks are:

- `math_props.ipynb` — elementary spectral properties;
- `univariate_fitting.ipynb` and `bivariate_fitting.ipynb` — synthetic scaling;
- `higgs_scaling.ipynb` and `criteo_scaling.ipynb` — real-data scaling;
- `higgs_robustness.ipynb` — feature-sensitivity histograms.

## Citation and license

Formal citation metadata will be added with the public paper record. Until then,
identify reproduced results by the repository URL and the producing commit in
the result manifest.

The code is released under the [BSD 3-Clause License](LICENSE).
