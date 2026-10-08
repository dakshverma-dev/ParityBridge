# Benchmark methodology

Read this before quoting any number from the repo.

## Protocol versions
New runs record `benchmark_protocol = median-paired-v2` in `meta.json`. The historical
`colab-gpu`, `colab-gpu-quick`, and `local-cpu` CSVs predate this correction and are
preserved as collected. Historical surface throughput selected the best of two runs;
qLDPC used one run. In CUDA surface runs, the CPU-only modes also synchronized CUDA.
Their near-1x performance ratios need a corrected GPU rerun before drawing conclusions.
Changing the harness does not retroactively validate those timings.

## Definitions (used everywhere)
* **Throughput ratio** = GPU-pipeline shots/s &divide; CPU-baseline shots/s on the same shots in the same session. &gt;1 means the GPU pipeline is faster, &lt;1 means it is slower. It is *not* a latency.
* **Latency ratio** (if quoted) = CPU latency &divide; GPU latency for the same batch size.
* **End-to-end** = stage 1 (pre-decoder) + stage 2 (global decoder), transfers included, GPU-synchronised.
* **Identical logical-error counts** on the same shots mean the two decoders made the same number of mistakes there; this is evidence, not a proof of statistical equivalence. Wilson 95 % intervals are in the CSVs and the detailed tables.

## Noise models
* **Surface code:** Stim `surface_code:rotated_memory_z`, `rounds = d`, uniform circuit-level depolarising/flip noise `p` on gates, resets, measurements and idle data qubits.
* **qLDPC:** bivariate-bicycle codes [[72,12,6]] and [[144,12,12]] under **code-capacity** i.i.d. bit-flip noise `p` (perfect syndrome measurement). This is a simplification and is labelled as such in every table.

## Data splits
Train / validation / test use different seeds (1 / 2 / 3). The learned gate is trained on train, its threshold calibrated on validation, and all reported numbers come from test.

## Metrics
| Metric | Definition |
|---|---|
| LER | logical failures / shots, with a 95 % Wilson interval (`ler_lo`, `ler_hi`) |
| Throughput | median of test shots / wall-clock seconds over 5 repetitions after warm-up (`--reps`, minimum 3) |
| Throughput dispersion | min/max and 25th/75th percentiles in `throughput_*_sps`; all durations in `throughput_times_s` (JSON array) |
| Paired throughput ratio | CPU time / candidate time per repetition; median, min/max and quartiles in `paired_ratio_*`; corresponding CPU durations in `paired_baseline_times_s` |
| Latency | per-shot wall-clock for batch size 1, p50/p95/p99 over `lat` shots after warm-up |
| `accept_rate` | surface: fraction of shots fully resolved before the global decoder; qLDPC: fraction where GPU BP converged |
| `syndrome_weight_kept` | residual syndrome weight / original weight after pre-decoding |

Decoder order alternates forward/reverse across throughput repetitions on identical
shots. Predictions must be identical across repetitions or the run fails. This is a
timing repeat, not an independent noise sample: LER is computed once on the test shots.
Stage timings are medians independently and need not sum to the median total.
CPU-only modes use CPU timers for throughput and both latency measurements; CUDA modes
synchronize their selected CUDA device. Surface throughput excludes diagnostic
bookkeeping, as in the original pipeline; transfers and prediction assembly are included.
qLDPC throughput wraps the complete decoder call, including fallback and bookkeeping.

CPU/local measurements finish before MLP training. The MLP uses a separate group of
fresh, alternating CPU/MLP repetitions; its paired CPU times are saved in its row.
Consequently its paired ratio need not equal the ratio to the earlier CPU summary row.
`--surface-modes none zero local` skips all MLP training and validation sampling.
Model initialization and minibatch order use fixed seeds; CUDA training is not promised
bitwise reproducible across hardware/library versions. Training stays outside decoding
timers and transfers only one host minibatch at a time.

## Baselines
* Surface: PyMatching (CPU) on the raw syndrome; plus a trivial non-AI shortcut (empty syndrome => no flip) so AI/GPU gains are not overstated.
* qLDPC: `ldpc.BpOsdDecoder` (C++), min-sum, 50 iterations, OSD-CS order 7, scaling 0.8.
* NVIDIA Ising surface full-pipeline: `ldpc.BpOsdDecoder` configured as min-sum, 30 iterations, OSD-0, scaling 0.8 for both the CPU baseline and CPU fallback path.

## Fairness rules
1. CPU and GPU numbers are compared **on the same machine/session** (Colab: 2 vCPU + T4). Do not mix numbers across machines.
2. The selected GPU, driver (when `nvidia-smi` is available), CUDA build, library versions,
   CPU/thread counts, git state, source hash, protocol and configuration are saved in
   `results/<tag>/meta.json`.
3. Batch-1 latency is reported separately from batched throughput. A GPU is expected to lose at batch size 1 for small codes; report where the crossover is.
4. LER parity is a precondition: a faster decoder with a worse LER is shown as a trade-off, not a win.
5. Statistics: low-LER points (d>=9) need many shots; if `errors` < ~30 treat the point as indicative only.
6. Report the accuracy/throughput trade-off together. The frontier figure shows measured
   LER difference versus throughput ratio for each radius, distance and physical error
   rate; it does not establish an optimal frontier or statistical equivalence.

## Known limitations
* CPU PyMatching is extremely fast at small distance; expect GPU benefit to appear at larger d and larger batches.
* Code-capacity qLDPC results do not transfer to circuit-level noise.
* The CUDA-Q QEC adapter has not been verified against a real install. The Ising adapter has been exercised with trained weights in the final validation run; reproducing those trained-weight runs requires a user-supplied Hugging Face token.
* The Ising comparison uses NVIDIA's circuit (25-parameter noise, boundary detectors, basis X), which differs from the plain Stim circuit used elsewhere, so those rows are kept in a separate table and never mixed.
* Dense BP memory scales as B x m x n.
