"""Full pipeline: NVIDIA Ising (GPU) -> GPU BP(+OSD) vs CPU baselines, total time + logical errors.

  python scripts/run_full_pipeline.py --repo third_party/Ising-Decoding --download --device cuda --tag colab-pipeline
(needs a clone of NVIDIA/Ising-Decoding and your own Hugging Face credentials; see docs/GPU_RUNBOOK.md)
"""
import argparse
import gc
import json
import traceback
from pathlib import Path

import pandas as pd

from qechybrid.bench import environment_meta, resolve_device
from qechybrid.ising_adapter import build_context, download_weights, run_full_pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="third_party/Ising-Decoding")
ap.add_argument("--weights", default=None)
ap.add_argument("--download", action="store_true")
ap.add_argument("--model-id", type=int, default=1)
ap.add_argument("--distances", type=int, nargs="+", default=[9, 13])
ap.add_argument("--ps", type=float, nargs="+", default=[0.003, 0.005])
ap.add_argument("--shots", type=int, default=20000)
ap.add_argument("--cpu-shots", type=int, default=1000, help="CPU BP+OSD-0 baselines run on this many shots only")
ap.add_argument("--device", default="auto")
ap.add_argument("--tag", default="pipeline")
a = ap.parse_args()

device = resolve_device(a.device)
weights = a.weights or (download_weights(a.model_id) if a.download else None)
out = Path("results") / a.tag
out.mkdir(parents=True, exist_ok=True)
meta = environment_meta(device)
meta["ising_weights"] = "trained" if weights else "RANDOM (plumbing check only)"
meta["cpu_bposd_shots"] = a.cpu_shots
(out / "meta.json").write_text(json.dumps(meta, indent=2))
rows = []
failed = False
for d in a.distances:
    for p in a.ps:
        try:
            ctx = build_context(a.repo, distance=d, p=p, shots=a.shots, model_id=a.model_id, weights=weights, device=device)
            rows += run_full_pipeline(ctx, cpu_shots=a.cpu_shots)
            pd.DataFrame(rows).to_csv(out / "pipeline.csv", index=False)
        except Exception:
            print(f"!! case d={d} p={p} failed:")
            traceback.print_exc()
            failed = True
        finally:
            ctx = None
            gc.collect()
            if device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
if failed:
    raise SystemExit("one or more full-pipeline benchmark cases failed; refusing to report success")
print("wrote", out / "pipeline.csv")
