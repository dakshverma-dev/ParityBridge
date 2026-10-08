# GPU runbook (Google Colab, free tier)

## Corrected measurements (any CUDA Python host)

Use a fresh tag so historical measurements remain intact. After installing this
branch and the appropriate CUDA PyTorch build, verify `torch.cuda.is_available()`
and start with the untrained local path:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/run_benchmarks.py --profile quick --device cuda --only surface --surface-modes none zero local --reps 5 --tag gpu-corrected-quick
python scripts/run_benchmarks.py --profile full --device cuda --only surface --surface-modes none zero local --reps 10 --tag gpu-corrected-full
python scripts/run_benchmarks.py --profile full --device cuda --only qldpc --reps 5 --tag gpu-corrected-qldpc
python scripts/make_report.py
```

Start with quick before spending a limited GPU session on full. The full profile
uses 200,000 surface shots per point; inspect error counts and Wilson intervals
before making accuracy claims. No fixed shot count guarantees enough errors at
low LER. The learned gate remains an optional ablation (`--surface-modes nn`).
Hugging Face/NVIDIA model access and a local CUDA PyTorch runtime are separate
requirements: the commands above need a host on which this Python code can run.

1. **Push the repo** to a public GitHub repository (see `PUBLISHING.md`).
2. Colab > *File > Open notebook > GitHub* > your repo > `notebooks/04_run_everything_colab.ipynb` (runs every step below; `00`-`03` are single-step subsets).
3. *Runtime > Change runtime type > T4 GPU*. Edit `GITHUB_REPO` in the first cell.
4. *Runtime > Run all*. Time: ~5 min with `--profile quick`, ~20-40 min with `full` (edit the benchmark cell).
5. The last cell downloads `results_colab.zip`. Unzip into the repo (`results/colab-gpu/`), then locally:
   ```
   python scripts/make_report.py
   git add results docs && git commit -m "Add Colab GPU results" && git push
   ```

## NVIDIA Ising head-to-head (a step of notebook 04; also notebook 03 alone)
1. On Hugging Face, open `nvidia/ising_decoder_surface_code_1_fast`, accept the terms, create a **read** token.
2. In Colab, add it as a secret named `HF_TOKEN` (key icon, notebook access on). Never paste it into a cell or commit it.
3. Run notebook 04 (or 03). NVIDIA validates the model for d >= 9, so the default distances are 9 and 13.
4. Without the token, `scripts/run_ising_bench.py` still runs with random weights: useful only to check plumbing and timing (rows are labelled and the LER is meaningless).

## Full pipeline (AI pre-decoder + GPU decoder vs CPU baselines)
`scripts/run_full_pipeline.py` (run by notebook 04 after the Ising step, tag `colab-pipeline`) measures, on identical shots, total time and logical errors for:
CPU PyMatching, CPU BP+OSD-0, NVIDIA Ising + PyMatching, Ising + CPU BP+OSD-0, and the full pipeline **Ising (GPU) -> GPU BP with CPU OSD-0 fallback**.
The CPU BP+OSD-0 rows use `--cpu-shots` (default 1000) because BP+OSD-0 costs milliseconds per shot on this circuit. For the publishable like-for-like benchmark, set `--cpu-shots` equal to `--shots` (the final d=9 validation used 20,000/20,000 shots).

## Troubleshooting
| Symptom | Fix |
|---|---|
| `CUDA: False` | Runtime type is CPU; switch to GPU and re-run all |
| Out of memory in BP | lower `chunk` in `BatchedMinSumBP` (default 2048) or use the [[72,12,6]] code |
| Out of memory for d=13 gate training | reduce `train` in `PROFILES["full"]` |
| Out of memory in `local_predecoder.py` at very large d | on CUDA the adjacency matrices are dense (N x N floats, ~5 MB at d=13); only d>30 gets large |
| `cudaq_qec` import fails | `pip install cudaq-qec`, restart runtime; otherwise the harness skips that backend and says so in `meta.json` |
| Session disconnects | results are written per configuration to `results/<tag>/*.csv`; re-run with `--only surface` / `--only qldpc` |
| Free quota exhausted | use Kaggle Notebooks (T4/P100, ~30 GPU h/week) with the same commands |

## Profiling for the demo (Nsight Systems)
Colab does not expose Nsight easily; for a timeline figure use PyTorch's profiler instead:
```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    hybrid.decode_batch(test_dets)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")   # open in chrome://tracing or ui.perfetto.dev
```
