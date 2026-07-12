# spectral_neuron_paper

Research code for the spectral neuron paper.

## Criteo scaling experiment

The experiment expects the headerless, tab-separated training file from the Criteo
Display Advertising Challenge. The first run builds a memory-mapped cache beside the
raw file; later runs reuse it.

```bash
uv run python -m paper.experiments.criteo_scaling \
  --data /path/to/train.txt \
  --profile sanity \
  --workers 2
```

Use `--cache-dir` to place the cache elsewhere, `--out` to override the raw CSV path,
and `--summary-out` to also write validation-selected aggregate results.
