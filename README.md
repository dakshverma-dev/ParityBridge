# qechybrid: GPU-accelerated hybrid quantum error-correction decoding

Quantum error correction needs fast classical decoding. This research repo measures
how much batch throughput a GPU pre-decoder can gain, and what decoding accuracy it costs.

```text
syndrome → parallel pre-decoder → residual syndrome → strong global decoder → prediction
```

* Historical T4 qLDPC runs reached 0.98–1.58× CPU batch throughput with equal logical-error counts on the tested shots. They used a single throughput repetition.
* Corrected local T600 full measurements show radius 2 slower at every surface point, with five additional errors across four points. Radius 1's marginal aggregate gains have paired timing ranges crossing 1× and add errors. Historical T4 timings still need a corrected T4 rerun.
* Single-shot latency favored the CPU on the measured small codes. Batch throughput is a separate metric.
* Trained Ising/full-pipeline performance remains unverified; the learned MLP is an optional ablation.

**Not claimed:** a real-time GPU advantage, MWPM equivalence of the local heuristic,
or statistically established accuracy equivalence from matching error counts.

![Corrected local surface accuracy and throughput trade-off](docs/figs/local-gpu-corrected-full_surface_frontier.png)

*Corrected T600 full run, 200,000 surface shots per point, five timing repetitions.
Aggregate throughput ratios in this figure differ from the median of paired ratios;
both are reported with timing spread in [the local results](docs/RESULTS_local-gpu-corrected-full.md).
The historical T4 quick run below has 72 vs 71 errors at one point and predates
the corrected [timing methodology](docs/METHODOLOGY.md).*

To reproduce the corrected local path, create/activate a virtual environment and
install the PyTorch build for your platform first (CPU example below). Then:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/run_benchmarks.py --profile quick --device auto --only surface --surface-modes none zero local --reps 5 --tag corrected-quick
```

For a CUDA host, preinstall the corresponding CUDA PyTorch build instead of the CPU
build. See [the GPU runbook](docs/GPU_RUNBOOK.md) for larger experiments and
[CONTRIBUTING.md](CONTRIBUTING.md) for the editable-install and lock-file workflow.
The historical results and detailed architecture follow below.

Validation of each finding, exact regression tests, and larger local measurements:
[docs/VALIDATION.md](docs/VALIDATION.md).

## Pipelines

Surface code (circuit-level noise, Stim). A pre-decoder removes the easy, local part of the syndrome; a global decoder handles what is left:

```
syndrome ──► pre-decoder (GPU) ──► residual syndrome ──► PyMatching (CPU, MWPM) ──► logical flip
              our local rule  or   (empty residual = done)
              NVIDIA Ising CNN
```

qLDPC codes (bivariate-bicycle codes, code-capacity noise). **Measured path:** batched BP on the GPU, with a CPU OSD fallback for the shots BP cannot resolve:

```
syndrome ──► batched min-sum BP (GPU) ──► converged? ──yes──► estimate
                                           │ no
                                           └──► those shots are copied to the CPU and decoded by `ldpc` BP+OSD
```

**How the fallback works.** BP runs on the GPU for every shot in the batch. A shot is *converged* when BP's estimate reproduces its syndrome; converged shots are accepted as they are. The remaining shots are copied to the CPU and decoded from scratch by the C++ `ldpc` BP+OSD decoder (it repeats BP and then applies OSD), one shot at a time. OSD itself is **not** on the GPU in this implementation, and the time spent in the CPU fallback is included in every reported throughput. The fraction of shots BP resolved on its own is recorded as `accept_rate` ("BP converged") in the qLDPC CSVs.

**Unverified alternative (never run here):** replacing that CPU fallback with NVIDIA's closed-source CUDA-Q QEC `nv-qldpc-decoder` on the GPU (`cudaq_qec_adapter.py`). `cudaq-qec` was not installed in any of our runs, so no result in this repository uses it.

**Experimental full pipeline** (measured on trained NVIDIA Ising weights): the same two ideas combined on the surface code,

```
syndrome ──► NVIDIA Ising (GPU) ──► residual syndrome ──► batched BP (GPU) ──► CPU `ldpc` BP+OSD on non-converged shots ──► logical flip XOR Ising's partial flip
```

*Why BP is valid on the surface code here:* Stim turns the noisy circuit into a **detector error model (DEM)**, i.e. a binary parity-check matrix **H** (rows = detectors, columns = fault mechanisms)
and an observable matrix **L** (rows = logical observables). Any decoder that accepts (H, syndrome) can decode it: BP+OSD estimates which faults fired (ê) and the logical outcome is **L·ê mod 2**.
The Ising model outputs a partial logical correction and a *residual* syndrome (the syndrome with the corrections it predicted removed); the residual is still a valid syndrome of the remaining faults under the same DEM,
so BP (using the undecomposed DEM) or PyMatching (using the decomposed, graph-like DEM) can finish the job. Final answer = Ising's partial flip XOR the global decoder's flip.
PyMatching stays the main surface-code baseline because it is the fastest practical CPU decoder; BP+OSD is run on the residual to test the GPU-decoder idea and to compare like with like against CPU BP+OSD.

## How to read the numbers

* **Throughput ratio** = GPU-pipeline shots/s ÷ CPU-baseline shots/s on the same shots in the same session. **Above 1: the GPU pipeline is faster. Below 1: it is slower.** It is not a latency.
* **Latency ratio** (where quoted) = CPU latency ÷ GPU latency for the same batch size.
* **End-to-end** = pre-decoder + global decoder, transfers included, GPU-synchronised.
* **Logical errors (ours / baseline)** are counts on the same shots. Equal counts mean both decoders made the same number of mistakes on those shots; they do not prove the decoders are equivalent. Wilson 95 % intervals are in the CSVs and in [docs/RESULTS.md](docs/RESULTS.md).
* **What `x` means:** a number written like `1.58x` is a throughput ratio as defined above (the `x` is only a unit symbol); `0.98x` means the GPU pipeline was slightly slower. Ranges such as `0.98-1.58` in the history are the same ratio without the symbol.

<!-- RESULTS:START -->
## GPU benchmark results (Tesla T4)

**Measured on Tesla T4 (Google Colab, 2 vCPU); qLDPC: `full` profile, 20,000 shots per point (`results/colab-gpu/`).**

*Throughput ratio = GPU-pipeline shots/s ÷ CPU-baseline shots/s on the same shots in the same session. A value above 1 means the GPU pipeline is faster, below 1 means it is slower. It is not a latency.*

* **qLDPC (BB codes, code-capacity noise): GPU/CPU batch-throughput ratio 0.98x to 1.58x** vs the C++ `ldpc` BP+OSD (best: BB[[144,12,12]], p=0.02, 1.58x; the GPU is slower at points below 1.00x). Logical-error counts are identical at every point on the same shots (not a proof of equivalence; intervals in the tables below).
  *This is batch throughput, not latency: single-shot latency is worse on the GPU than on the CPU, so this is not a real-time result.*
* **Surface code, historical radius-1/2 code, quick profile (d=5, 7; 20,000 shots; `results/colab-gpu-quick/`), conservative r=2 rule: throughput ratio 1.08x to 1.32x** (4 of 4 points above 1.02x; 1 extra logical error in total at the points where ours was worse).
* **Surface code, historical radius-1/2 code, quick profile (d=5, 7; 20,000 shots; `results/colab-gpu-quick/`), fast r=1 rule: throughput ratio 0.90x to 2.53x** (3 of 4 points above 1.02x; 12 extra logical errors in total at the points where ours was worse).
* **Surface code, older code, full profile (d=5 to 13; 200,000 shots; `results/colab-gpu/`): throughput ratio 0.62x to 1.19x** (4 of 12 points above 1.02x; 344 extra logical errors in total at the points where ours was worse).

| code | p | C++ ldpc BP+OSD (shots/s) | GPU BP + OSD fallback (shots/s) | throughput ratio | logical errors (GPU / ldpc) |
|---|---|---|---|---|---|
| BB[[72,12,6]] | 0.02 | 77,638 | 86,407 | **1.11x** | 226 / 226 |
| BB[[72,12,6]] | 0.04 | 39,769 | 38,941 | **0.98x** | 1770 / 1770 |
| BB[[72,12,6]] | 0.06 | 18,921 | 21,240 | **1.12x** | 4991 / 4991 |
| BB[[144,12,12]] | 0.02 | 39,426 | 62,340 | **1.58x** | 13 / 13 |
| BB[[144,12,12]] | 0.04 | 19,885 | 24,982 | **1.26x** | 233 / 233 |
| BB[[144,12,12]] | 0.06 | 7,627 | 7,853 | **1.03x** | 1772 / 1772 |

Historical timing protocol: surface used best-of-two, qLDPC used one repetition, and CPU-only surface modes synchronized CUDA in GPU runs. Marginal surface ratios need a corrected GPU rerun. Caveats: batch throughput only (the GPU is slower than the CPU for single shots; see `docs/RESULTS.md`). Historical surface-code results for both radii come from the quick profile only (d=5 and 7, small error counts); the older full-profile surface-code result (last bullet above) used the first, aggressive radius-1 rule and a slower stage 1. Ratios are relative to CPU baselines on the same Colab machine.

Confidence intervals, surface-code tables and the environment record: [docs/RESULTS.md](docs/RESULTS.md), [docs/RESULTS_colab-gpu-quick.md](docs/RESULTS_colab-gpu-quick.md).
<!-- RESULTS:END -->

## Status at a glance

| Component | Status |
|---|---|
| Batched GPU BP + OSD fallback (qLDPC) | **Measured on a T4** (full profile, 20k shots/point): see the table above |
| Local surface-code pre-decoder, first aggressive radius-1 rule | **Measured on a T4** (full profile, 200k shots): throughput ratio 0.62x-1.19x and extra logical errors at some points |
| Local pre-decoder: conservative radius-2 rule and `fast` radius-1 rule, with the fp16 stage 1 | **Measured on a T4, quick profile only** (d=5 and 7, 20k shots) |
| Learned MLP gate | **Measured on a T4**: resolved at most ~2 % of shots at d>=7, so it was not useful there; kept as an ablation |
| NVIDIA Ising trained model | **Measured with trained weights** at d=9 across p=0.001, 0.003, 0.005 and 0.010; residual-syndrome and fallback statistics recorded |
| CUDA-Q QEC `nv-qldpc-decoder` | **Not verified**: `cudaq-qec` was not installed in any run (the decoder is a closed-source library, see the [CUDA-Q QEC docs](https://nvidia.github.io/cudaq-qec/)) |
| Full pipeline: Ising (GPU) -> GPU BP+OSD vs CPU baselines | **Measured** at d=9 with trained Ising weights; 20k-shot benchmark and paired correctness audits committed under `results/` |

## Glossary

* **PyMatching / MWPM:** minimum-weight perfect matching on a graph of detectors; the standard fast surface-code decoder.
* **BP (belief propagation):** iterative message passing on the parity-check graph; fast and parallel, but it can fail to converge on short cycles.
* **OSD (ordered-statistics decoding):** a linear-algebra post-processing step that rescues shots where BP does not converge.
* **Ising pre-decoder:** NVIDIA's learned 3D-CNN that removes easy local errors from the syndrome before a global decoder runs ([repo](https://github.com/NVIDIA/Ising-Decoding)).
* **Stim / DEM:** circuit simulator and the detector error model it derives (see above).
* **qLDPC / bivariate-bicycle (BB) code:** a high-rate quantum code family; we use [[72,12,6]] and [[144,12,12]].
* **Code-capacity vs circuit-level noise:** code-capacity assumes perfect syndrome measurement (simpler, used for the qLDPC codes); circuit-level models noisy gates and measurements (used for the surface code).

## What is new here, what is not

**Our contribution is the combination and the measurement**: a GPU-batched BP implementation, a GPU surface-code pre-decoder with an accuracy/throughput dial, and a reproducible, identical-shot comparison framework
around NVIDIA's Ising pre-decoder, with explicit throughput, latency and logical-error measurements, including the negative results.

**Not ours (used as-is, credited):** NVIDIA's [Ising pre-decoder and weights](https://github.com/NVIDIA/Ising-Decoding) (NVIDIA Open Model License; code Apache-2.0),
[CUDA-Q QEC](https://github.com/NVIDIA/cudaqx) ([docs](https://nvidia.github.io/cudaq-qec/)), [Stim](https://github.com/quantumlib/Stim), [PyMatching](https://github.com/oscarhiggott/PyMatching),
[`ldpc`](https://github.com/quantumgizmos/ldpc), PyTorch.

**Ours:**
* a batched, edge-list **GPU belief-propagation decoder** that drops converged shots from the working batch (unit-tested to match a dense reference bit for bit; equal logical-error counts to `ldpc` on a T4),
* a **local GPU pre-decoder for the surface code** with a measured accuracy/throughput dial (`radius=2` had one extra error at one point in the stored quick T4 run, `radius=1` is faster but adds errors at some points); in the one measured case (d=9, p=0.003, NVIDIA's circuit) it leaves substantially more syndrome weight than the Ising model (48 % vs 2.9 %),
* the **benchmark harness** (same shots, same session, Wilson intervals, stage timings, single-shot and 256-shot micro-batch latency) and the **head-to-head wiring around NVIDIA's pipeline**,
* the reproducible Colab notebook and results site.

**What we claim (and only this):**
1. GPU BP gives equal logical-error counts to `ldpc` on the same shots and a GPU/CPU batch-throughput ratio of 0.98x-1.58x on a T4 (qLDPC, code-capacity noise; best case [[144,12,12]], p=0.02).
2. On a T4 (run 3, quick profile, d=5 and 7, 20k shots) the conservative radius-2 rule has a throughput ratio of 1.08x-1.32x with 72 vs 71 logical errors at d=7, p=0.004 and equal counts at the other three points; the fast `radius=1` rule reaches 2.53x (d=7, p=0.002) with 80 vs 71 errors at d=7, p=0.004.
   This is a small quick-profile result; the earlier full run used the first aggressive rule and a slower stage 1 (0.62x-1.19x).
3. In one case (d=9, p=0.003) NVIDIA's trained Ising model left 2.9 % of the syndrome weight and resolved 47 % of shots completely. Its end-to-end timing has not been validly measured.

**What we do not claim:** a real-time (single-shot, microsecond) GPU advantage (in run 3 the GPU surface-code path took 630-880 µs per single shot vs 18-38 µs for PyMatching, and was slower at 6 of 8 points at 256-shot micro-batches); any Ising or full-pipeline throughput result
(`results/colab-ising/` and `results/colab-pipeline/` do not exist yet); equivalence of the local rule to MWPM (it is a heuristic).

### The full-pipeline experiment (`scripts/run_full_pipeline.py`): measured

On identical shots from NVIDIA's circuit with trained Ising weights, the experiment measures total wall-clock time and logical errors for CPU PyMatching, CPU BP+OSD, Ising + PyMatching, Ising + CPU BP+OSD, and the full Ising (GPU) -> GPU BP with CPU OSD fallback pipeline.

At d=9, p=0.003 and 20,000 shots, CPU BP+OSD took 9,627.4 us/shot and the complete hybrid pipeline took 6,176.6 us/shot, giving a 1.56x speedup for the like-for-like BP+OSD workload. PyMatching remained substantially faster, so the hybrid is not presented as a replacement for PyMatching.

Stress tests at p=0.001, 0.003, 0.005 and 0.010 recorded increasing CPU fallback fractions of 12.5%, 37.2%, 62.1% and 96.3%, respectively.

Paired correctness audits at p=0.001, 0.003 and 0.005 used 20,000 identical shots per point and found zero shot-level disagreements between Ising + CPU BP+OSD and the GPU hybrid decoder (60,000/60,000 agreement).

## What is in the repo

| Piece | File | Tests |
|---|---|---|
| Circuit-level surface-code data + detector error model (Stim) | `src/qechybrid/data.py` | unit-tested |
| Local GPU pre-decoder (clique-style heuristic, not provably MWPM-equivalent) | `local_predecoder.py` | unit-tested; CPU/GPU code paths checked for equality |
| Learned MLP gate | `gate.py` | unit-tested |
| Hybrid surface-code pipeline + trivial non-AI baseline | `pipeline.py` | unit-tested |
| Batched min-sum BP (CUDA/CPU) with BP+OSD fallback; circuit-level `DemBpOsd` | `bp_gpu.py`, `decoders.py` | unit-tested against a dense reference |
| Bivariate-bicycle qLDPC codes, GF(2) row-space logical-failure check | `codes.py`, `gf2.py` | unit-tested |
| Optional CUDA-Q QEC `nv-qldpc-decoder` backend | `cudaq_qec_adapter.py` | not tested (library not installed) |
| NVIDIA Ising head-to-head and full pipeline | `ising_adapter.py`, `scripts/run_ising_bench.py`, `scripts/run_full_pipeline.py` | trained-weight benchmark + paired correctness audits |
| Benchmark harness (LER with Wilson CIs, throughput, p50/p95/p99 single-shot latency, 256-shot micro-batch latency, environment record) | `bench.py`, `scripts/run_benchmarks.py` | unit-tested |
| Results website (GitHub Pages ready) | `docs/index.html` (built by `scripts/make_report.py`) | element-id test |

### Results layout
`results/<tag>/` holds data only: `colab-gpu` (historical T4, full profile), `colab-gpu-quick` (historical T4, quick profile), `local-cpu` (historical CPU), and new corrected CPU/T600 quick/full runs. Ising/pipeline results are stored separately when available.
Run notes live in `docs/runs/` (`colab-gpu-run1.md`, `colab-ising-run1.md`). From now on every `meta.json` also records the git commit, `ldpc` version, CUDA version and GPU memory;
older runs lack some of these and `docs/RESULTS.md` says "not recorded".

## Documentation

| Doc | Contents |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md), [docs/RESULTS_colab-gpu-quick.md](docs/RESULTS_colab-gpu-quick.md) | full generated tables with confidence intervals and the environment record |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | pipeline, modules, design decisions, extension points |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | definitions, noise models, metrics, baselines, fairness rules, limitations |
| [docs/GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md) | running on Colab, troubleshooting, profiling |
| [docs/PUBLISHING.md](docs/PUBLISHING.md) | public repo + GitHub Pages checklist |
| [docs/PITCH.md](docs/PITCH.md) | 3-minute pitch outline |
| [CONTRIBUTING.md](CONTRIBUTING.md) | dev workflow; `make test / quick / full / report` also work |

## Quick start (local, CPU, virtual environment)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1     # Windows: creates .venv, installs, runs tests
# or: bash scripts/setup.sh                                    # Linux / macOS / WSL
.\.venv\Scripts\python scripts/run_benchmarks.py --profile quick --device cpu --tag local-cpu
.\.venv\Scripts\python scripts/make_report.py                  # -> docs/index.html
```

## GPU run (free: Google Colab T4)

**Use one notebook: `notebooks/04_run_everything_colab.ipynb`.** It runs everything below in order with *Runtime > Run all*.

| notebook | purpose |
|---|---|
| **`04_run_everything_colab`** | **setup check, tests, surface + qLDPC benchmarks, Ising head-to-head, full pipeline, report + zip (use this one)** |
| `00_setup_check_colab` | subset: GPU/install/tests only |
| `01_run_benchmarks_colab` | subset: surface + qLDPC benchmarks only |
| `02_cudaq_qec_bposd` | subset: optional CUDA-Q QEC backend (unverified) |
| `03_ising_head_to_head` | subset: Ising comparison only (does not include the full pipeline) |

1. Push this repo to a **public** GitHub repository and open notebook 04 from GitHub in Colab (runtime: an NVIDIA GPU; set `GITHUB_REPO` and `PROFILE` in the first cell).
2. For the Ising and full-pipeline steps you need a free Hugging Face account, acceptance of the model terms, and a *read* token stored as the Colab secret `HF_TOKEN` (never paste it into a cell). Without it those steps are skipped.
3. Download the zip, copy the `results/colab-*` folders into the repo, run `python scripts/make_report.py` and `python scripts/make_summary.py` (and `... colab-gpu-quick`), commit, and enable **GitHub Pages** (Settings > Pages > `main` / `docs`).

## Results history (short)

* **Run 1 (first code):** GPU BP was slower than the C++ `ldpc` (ratio 0.07-0.34); the first surface-code pre-decoder reached at most 1.25.
* **Run 2 (full profile, BP rewrite):** qLDPC ratio 0.98-1.58 with equal logical-error counts; the first (aggressive) surface-code rule 0.62-1.19 with extra logical errors at some points.
* **Ising head-to-head:** the original trained-weight timing at d=9, p=0.003 was invalid because compile/warm-up occurred inside the timed region. The corrected implementation now warms up before timing and measures the complete trained-Ising pipeline separately; the corrected d=9, p=0.003 benchmark leaves 2.94 % residual syndrome weight.
* **Run 3 (historical radius-1/2 code, quick profile):** the conservative radius-2 rule 1.08-1.32 with one extra logical error at one point; the `fast` radius-1 rule 0.90-2.53 with extra errors at one point.

Full chronology with the numbers behind each statement: [docs/HISTORY.md](docs/HISTORY.md).

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Notes

* Noise: circuit-level (Stim `rotated_memory_z`) for the surface code; code-capacity bit-flip noise for the qLDPC codes (stated wherever results are shown). The Ising comparison uses NVIDIA's own circuit and noise model and is kept in separate tables.
* NVIDIA Ising weights are under the NVIDIA Open Model License and are downloaded at run time, never committed.
* License: MIT for ParityBridge; see [LICENSE](LICENSE). The original ccn copyright and permission notice is preserved in [LICENSES/MIT-ccn.txt](LICENSES/MIT-ccn.txt); see [NOTICE](NOTICE) for attribution. Third-party code and model weights remain subject to their own licenses.
