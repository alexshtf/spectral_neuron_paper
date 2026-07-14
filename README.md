# spectral_neuron_paper

Research code for the spectral neuron paper.

## Criteo scaling experiment

The experiment expects the headerless, tab-separated training file from the Criteo
Display Advertising Challenge. The first run builds a memory-mapped cache beside the
raw file; later runs reuse it.

Feature preprocessing is fitted once on a reproducible 10% sample of the chronological
training split. Each model then makes one pass over a fixed random permutation of that
split, with validation measurements taken at every requested training-size checkpoint.

```bash
uv run python -m paper.experiments.criteo_scaling \
  --data /path/to/train.txt \
  --profile sanity \
  --workers 2
```

Use `--cache-dir` to place the cache elsewhere, `--out` to override the raw CSV path,
and `--summary-out` to also write validation-selected aggregate results.
