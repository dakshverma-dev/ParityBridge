# Local feedback validation and fixes

Validated against the checkout based on commit `1e0c610`; local work is on
`codex/benchmark-validation`. Historical result CSVs and metadata were preserved.

| Feedback | Validation | Local change |
|---|---|---|
| CPU surface baselines synchronize CUDA | Confirmed: `run_surface` passed CUDA into CPU-only pipelines and both latency timers. A regression test failed on the actual synchronization call. | CPU-only modes force CPU timing internally; the harness also uses CPU latency timers. CUDA timers synchronize the selected device. |
| Best-of-two throughput | Confirmed for surface. qLDPC actually used just one timed repetition. | Both now use five repetitions by default, alternating decoder order; median, min/max, quartiles, all durations and paired ratios are saved. |
| Machine-specific lock entry | Confirmed: `-e e:\cnc`. | Regenerated from the tested Windows/CPython 3.13.7 environment, excluding the project and platform-specific Torch wheel. Installation instructions and constraints are in its header. |
| Torch installation ambiguity | Confirmed: package metadata requires Torch, while platform selection was explained only in setup files. | README, requirements, contributing guide and runbook explicitly require choosing the Torch build first, then installing `.[dev]`. Torch remains a real runtime dependency. |
| Fresh checkout cannot run pytest | Reproduced `ModuleNotFoundError: qechybrid` before installation. | Canonical editable installation is prominent. No pytest-only path workaround was added. Existing CI already installed the package. |
| Broad scientific test bands | Confirmed; these were smoke checks, not meaningful regression bounds. | Marked them as slower quality smoke checks; added frozen small inputs, complete expected predictions and exact error counts. Added independent residual/flip invariants and real CUDA/reference equality tests. |
| Empty BP batch crashes | Reproduced in both edge-list and dense implementations. | Correct shapes/dtypes for NumPy and device-resident tensor outputs; hybrid empty batches avoid fallback. |
| Gate trains before every surface benchmark | Confirmed. | CPU/local measurements finish and checkpoint before MLP training. Mode selection can skip the gate entirely. Only one host minibatch is uploaded for training. Initialization and minibatch ordering are seeded. |
| Stale pre-decoder docstring / “safe” | Confirmed. | Updated the comment and renamed new radius-2 rows to “conservative”; historical raw row names remain unchanged. |
| Radius 2 had no added errors in the saved quick run | This part of the feedback was inaccurate. The stored T4 CSV has **72 vs 71** at d=7, p=0.004, with equal counts at the other three points. | Corrected README and summary-generator claims. Neither rule has an accuracy guarantee. |
| Frontier and landing-page presentation | Recommendations, rather than code defects. | Added static and interactive accuracy/throughput trade-off plots, a compact README introduction, explicit non-claims and an early reproduction command. |
| Trained Ising/full-pipeline performance | Final d=9, p=0.003 validation now exists with trained weights, plus stress tests and paired audits. | Added the measured pipeline results, corrected the full-pipeline timing boundary, and explicit BP+OSD-0 labeling; README avoids universal or PyMatching-superiority claims. |

## Verification

The first targeted run failed on the empty-batch and CPU/CUDA timer regressions.
After fixing them, the complete CUDA-enabled suite passed **36 tests**, with one
skip because NVIDIA's external Ising repository is not cloned. This includes real
CUDA tests, not just CPU emulation of the GPU algorithm. The CPU-only environment
also passes, with CUDA-specific tests skipped. `pip check`, `compileall`, and
`git diff --check` pass.

The rebuilt report was checked in the local browser: the corrected protocol label,
trade-off plot, run selector and paired-ratio columns render, with no console errors.
The MLP benchmark path was exercised separately with a small CPU training set; that
was a plumbing check, not a scientific quality measurement.

The scientific fixtures record versions and seeds in `tests/fixtures/meta.json`:
surface d=5/p=0.004 has 29 baseline errors, 27 radius-1 errors and 29 radius-2 errors
on 4,096 fixed shots; BB72/p=0.04 has 100 errors in both decoders on 1,024 fixed
shots. Tests compare entire predictions and syndrome validity, not just these totals.
Intentional fixture regeneration uses `scripts/update_regression_fixtures.py` and
requires reviewing changed predictions/counts. Tests never regenerate expectations.

## Corrected local measurements

The machine has an NVIDIA T600 Laptop GPU, 4 GB VRAM, driver 596.71; the CUDA
environment uses PyTorch 2.14.1+cu130. PyTorch CPU threads were set to 2 for these
benchmark sessions and recorded in metadata. Each decoder uses five repetitions.
The CPU environment is kept in `.venv`; CUDA is isolated in `.venv-cuda`.

* `results/local-cpu-corrected`: quick surface and qLDPC measurements on CPU.
* `results/local-gpu-corrected-quick`: completed CUDA quick surface and qLDPC run.
  The surface radius-2 aggregate throughput ratio is 0.50–0.74× PyMatching, radius 1
  is 0.66–0.94×. Radius 2 matches baseline error counts on these 20,000-shot points;
  radius 1 adds errors. This is evidence against a throughput win on this machine
  in this configuration, not a verdict on T4 performance.
* `results/local-gpu-corrected-full`: completed 12 surface points (d=5, 7, 9, 13;
  p=0.001, 0.002, 0.004; 200,000 shots each) and six qLDPC points (20,000 shots each).
  Radius 2 reached 0.09–0.81× PyMatching and added five errors across four points:
  883/882, 1/0, 449/448 and 118/116. Radius 1 added 418 errors across the points where
  it was worse. Its two aggregate ratios above 1× were 1.03× and 1.12×, but the
  corresponding paired medians were 1.016× and 0.964×; both paired min/max ranges
  crossed 1×. This is insufficient evidence of a robust throughput win.
  qLDPC reached 0.59–0.87× CPU throughput with matching error counts at all six points.
  The d=13 work approached this laptop's 4 GB device-memory limit; these timings
  describe this hardware and configuration, not a hardware-independent scaling law.

Full tables, Wilson intervals, repetition ranges and environment records are in
[the full local report](RESULTS_local-gpu-corrected-full.md); figures and the
interactive report are generated from the raw CSVs. Low-error points remain
indicative even with 200,000 shots. The source hashes in all corrected run metadata
match this branch's decoder source. Runs were collected before the local commit,
so their recorded base commit and dirty flag describe the state at measurement time.

All runs use identical shots between decoders within a configuration. Timing
repetitions are not independent noise samples, so their shot counts must not be
multiplied to compute LER uncertainty. Paired timing spread is not an accuracy
confidence interval. The trade-off plot does not establish an optimal Pareto
frontier, and matching error counts do not prove decoder equivalence.

The T600 run cannot validate the old T4 ratios across hardware. A corrected T4 or
owner-provided CUDA-host rerun remains necessary before restoring those marginal
speed claims. Commands for that rerun are in `GPU_RUNBOOK.md`. No remote model/API
compute or private model credentials were used for this local validation.
