# Architecture

## Pipeline

```
 syndromes (B x N, uint8)
        │
        ▼
 ┌───────────────────────────┐   residual all zero      ┌───────────────┐
 │ Stage 1: pre-decoder      │ ───────────────────────► │ prediction    │
 │  (GPU, batched)           │                          └───────────────┘
 └───────────┬───────────────┘
             │ residual syndrome (sparser) + partial logical flip
             ▼
 ┌───────────────────────────┐
 │ Stage 2: global decoder   │ ──► prediction = partial flip XOR global flip
 │  decoder (CPU or GPU)     │
 └───────────────────────────┘
```

Two instantiations share the same idea (cheap parallel stage first, the global decoder only for what is left):

| | Surface code (circuit-level noise) | qLDPC (bivariate bicycle, code-capacity noise) |
|---|---|---|
| Stage 1 | `LocalPreDecoder` (or `Gate` MLP) | `BatchedMinSumBP` (PyTorch) |
| "Easy" means | residual syndrome empty | BP converged to a syndrome-consistent estimate |
| Stage 2 | PyMatching on the residual | CPU `ldpc` BP+OSD-0 on non-converged shots in the measured full pipeline; CUDA-Q QEC GPU decoder (never run here) |
| Module | `pipeline.HybridDecoder` | `decoders.HybridBpOsd` |

## Modules (`src/qechybrid/`)

| Module | Responsibility |
|---|---|
| `data.py` | Stim rotated surface-code circuits, detector sampling, detector-error-model to (H, L, priors) |
| `local_predecoder.py` | Sparse-matrix local rule: mutually isolated fired pairs are matched along their edge if that is no costlier than two boundary matches |
| `gate.py` | MLP predicting the logical flip; threshold on \|logit\| calibrated on held-out data to limit extra logical errors (empirical, not a guarantee) |
| `pipeline.py` | `HybridDecoder` with modes `none`, `zero`, `nn`, `local`; batch and single-shot paths |
| `bp_gpu.py` | Edge-list batched normalised min-sum BP that drops converged shots from the working batch (`BatchedMinSumBP`); the dense original is kept as `DenseMinSumBP` for equivalence tests |
| `decoders.py` | `MatchingDecoder`, `CpuBpOsd`, `HybridBpOsd` (GPU BP + CPU OSD fallback), `DemBpOsd` (the same on a circuit-level detector error model) |
| `codes.py`, `gf2.py` | BB codes, GF(2) RREF, logical-failure test (is the residual in the row space of Hx?) |
| `cudaq_qec_adapter.py` | Optional NVIDIA CUDA-Q QEC backend (unverified: never run) |
| `bench.py`, `timing.py` | Benchmark profiles, latency percentiles, Wilson intervals, environment metadata |

## Design decisions

* **Local rule over a global classifier.** The MLP gate must be right about the whole syndrome volume; in our T4 runs it resolved at most ~2 % of shots at d>=7,
  whereas the local rule still fully resolved some shots there (e.g. 12-18 % at d=7, p=0.002). We have not shown why, so treat this as an observation.
* **Stage 2 is a standard decoder; Stage 1 is a heuristic.** The residual is decoded by PyMatching (or BP+OSD); any extra logical errors relative to PyMatching alone come from Stage 1's decisions,
  which is what the benchmarks compare. The pairing rule is not provably equivalent to MWPM: `radius=1` added errors at some points, `radius=2` showed none in our tests. A boundary rule exists (`use_boundary=True`) but is off by default because it measurably hurt LER.
* **Calibrated on validation data, not guaranteed.** The gate's threshold is chosen on a separate validation set so that it adds at most 5 % extra errors relative to MWPM there; on test data this did not always hold (d=7, p=0.001: 20 vs 5 logical errors).
* **Edge-list BP with active-set shrinking.** Messages live on the edges (padded to the maximum check/variable degree) and shots drop out of the working batch as they converge, so one hard shot does not keep the whole batch iterating.
  It was unit-tested against the dense reference for identical output. Memory scales with batch x checks x maximum check degree; `chunk` is chosen automatically for large irregular matrices.
* **Honest timing.** GPU stages are timed with `torch.cuda.synchronize()`. The measured full-pipeline path keeps the Ising residual on the GPU, includes GPU BP and CPU-fallback wall time, and includes the host transfer of the partial logical bit inside the full-pipeline timed region. Latency is single-shot.

## Extension points

* **NVIDIA Ising CNN as Stage 1:** implemented in `ising_adapter.py`. It wraps NVIDIA's `PreDecoderMemoryEvalModule` (flat detector bits in, `[logical flip | residual detectors]` out) and runs it next to plain PyMatching and our pre-decoder on the same shots. NVIDIA's repository is cloned at run time (Apache-2.0, not vendored); the weights are gated on Hugging Face and loaded with the user's own token.
* **CUDA-Q QEC OSD (unverified):** pass `osd_backend=CudaqQecBpOsd(H)` to `HybridBpOsd` to use NVIDIA's GPU BP+OSD for the fallback shots. It has never been run here.
* **Other codes:** any Stim circuit works for the surface-code path; any binary parity-check matrix works for the BP path.
