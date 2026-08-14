# Retained results

These frozen artifacts are the inputs used by the publication notebooks.
They are preserved from the listed producing commits; check out that commit for
byte-for-byte reproduction with the corresponding experiment code.

| Artifact | Profile | Producing commit | Rows | Source dataset identity |
|---|---|---:|---:|---|
| `univariate_full.csv.zst` | univariate `full` | `184d878` | 539,136 | Synthetic; no external dataset |
| `bivariate_full.csv.zst` | bivariate `full` | `de778f4` | 539,136 | Synthetic; no external dataset |
| `criteo_scaling_full_repeated_shuffle.csv` | Criteo scaling `full` | `16db39b` (run), `31ec192` (labels) | 8,712 | Criteo Display Advertising Challenge `train.txt` |
| `higgs_scaling_full_repeated_shuffle.csv` | HIGGS scaling `full` | `223de70` | 11,440 | HIGGS `HIGGS.csv`, 11,000,000 rows |
| `higgs_robustness_full_noise_0p5_repeated_shuffle.csv.zst` | HIGGS robustness `full`, noise 0.5 | `87c81f1` | 32,256 | HIGGS test split and the retained HIGGS scaling result |
