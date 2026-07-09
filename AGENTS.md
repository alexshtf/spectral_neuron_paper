# AGENTS.md

## Project Intent

This repo is research code for spectral neurons: scalar units of the form `x -> lambda_k(A_theta(x))`, where `lambda_k` is one selected eigenvalue and the generic matrix path is `A_theta(x) = A_0 + x_1 A_1 + ... + x_n A_n` with learned symmetric matrices `A_0, A_1, ..., A_n`.

Optimize for clear, minimal experiment infrastructure that supports the paper. Do not turn it into a general framework.

Prefer modern, explicit Python when it improves clarity. Keep tensor code vectorized and shape-aware; do not replace vectorized numerical work with Python loops unless the problem is genuinely tiny and clarity wins.

## Experiment Infrastructure

- Keep notebooks thin. Put reusable logic in `src/paper/`; notebooks should run, summarize, and plot.
- Prefer profiles over many CLI flags for experiment grids.
- Use FitStream-style event streams for small in-memory experiments. Reach for DataLoader only when scale, streaming, multiprocessing, or dataset complexity justifies it.
- Raw logs may be denormalized, but every column should have a clear role: run identity, selection, summary, or diagnostics.

## PyTorch and NumPy Boundaries

- Targets consume and return NumPy arrays.
- Tasks own NumPy-to-Torch conversion.
- Models and training code should operate on tensors only.
- Do not call `.numpy()` inside model or training paths.
- Do not treat NumPy arrays as tensors or tensors as NumPy arrays.

PyTorch modules should follow feature-last input convention:

```text
x.shape == (..., input_dim)
```

For univariate models, require `(..., 1)` rather than silently accepting ambiguous `(batch,)` inputs. Preserve leading dimensions in model outputs.

Use `[..., eig_idx]` for eigenvalue selection unless there is an intentional fixed-rank tensor contract.

Avoid hidden global RNG side effects. For deterministic model initialization, scope `torch.manual_seed(...)` with `torch.random.fork_rng()`. Use `np.random.default_rng(seed)` for NumPy sampling.

## Synthetic Data

For synthetic univariate tasks:

- training inputs are random uniform samples;
- training labels may include optional noise;
- validation may be random unless the user asks otherwise;
- test inputs must be linearly spaced over `[lower, upper]`;
- test labels must be exact/noiseless function values.

The synthetic test set measures function-fitting quality, not random-sample performance.

## Tuning and Metrics

- Select checkpoints using validation metrics only.
- Select learning rates using validation metrics only.
- Use test metrics only for final reporting.
- Aggregate across seeds with medians and quantiles, not minima.
- Prefer pandas groupby/sort/vectorized operations over row-wise loops.
- When evaluating validation/test metrics, switch models to `eval()` and restore their previous training mode afterward.

## Tests

Keep tests useful and sparse.

Good tests protect:

- model math and shape contracts;
- target properties such as reproducibility, monotonicity, and convexity;
- validation-vs-test leakage boundaries;
- seed aggregation rules;
- small end-to-end wiring that would catch a broken experiment path.

Avoid tests that only restate an obvious implementation line, constant, or config choice. Do not add focused regression tests for every cleanup unless the behavior is subtle or likely to regress.

## Verification

Use the repo-declared test dependency; do not install pytest ad hoc.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall src tests
git diff --check
```

Before committing staged changes, also run:

```bash
git diff --cached --check
```

## Git Scope

Follow the user's exact commit scope.

- "Commit current changes" means stage only the intended current work.
- "Commit all changes" means stage the whole tree.
- "Commit and push" means do both.

For notebook-heavy diffs, inspect code cells structurally rather than relying on raw `.ipynb` output noise.

## Code guideline
- Simplicity, conciseness, elegance, beauty, "taste", consistent formatting, and good engineering are of paramount importance.
- Use the allowed Python version, standard library and installed libraries to the maximum extent possible to achieve the above.
- The above guidelines are not at the expense of correctness or performance, unless performance impact is negligible or trivial.
