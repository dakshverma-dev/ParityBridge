"""Generate the Colab notebooks (run once: python scripts/make_notebooks.py)."""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent.parent / "notebooks"
OUT.mkdir(exist_ok=True)

SETUP = '''# ---- edit this: your public GitHub repo ----
GITHUB_REPO = "Rhytam23/ccn"
import os, subprocess, sys
if not os.path.exists("/content/repo"):
    subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/{GITHUB_REPO}.git", "/content/repo"], check=True)
%cd /content/repo
!pip -q install -r requirements-colab.txt
!pip -q install -e . --no-deps   # for shell commands (pytest, scripts)
sys.path.insert(0, "/content/repo/src")  # for this kernel: editable installs are not visible to an already-running kernel
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv'''


def nb(name, cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata["accelerator"] = "GPU"
    n.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
    nbf.write(n, OUT / name)


nb("00_setup_check_colab.ipynb", [
    ("md", "# 00 - Colab setup check\n**Runtime > Change runtime type > T4 GPU** first. Verifies the GPU, installs the package, runs the unit tests and checks whether NVIDIA CUDA-Q QEC is importable."),
    ("code", SETUP),
    ("code", "!pip -q install pytest\n!pytest -q"),
    ("code", '''from qechybrid import cudaq_qec_adapter
print("cudaq_qec:", cudaq_qec_adapter.available())
# Optional (Linux + NVIDIA GPU): !pip -q install cudaq-qec   # then restart runtime and re-run this cell'''),
])

nb("01_run_benchmarks_colab.ipynb", [
    ("md", "# 01 - Full GPU benchmark\nRuns the **full** profile on the Colab GPU (about 20-40 min on a T4; use `--profile quick` for a 5 min smoke test) and rebuilds the website. Download the zip or commit `results/` + `docs/index.html` to GitHub."),
    ("code", SETUP),
    ("code", "!python scripts/run_benchmarks.py --profile full --device cuda --tag colab-gpu"),
    ("code", "!python scripts/make_report.py\n!zip -qr results_colab.zip results docs/index.html docs/figs\nfrom google.colab import files; files.download('results_colab.zip')"),
])

nb("02_cudaq_qec_bposd.ipynb", [
    ("md", "# 02 - qLDPC: CUDA-Q QEC GPU BP+OSD vs our hybrid\nIf `cudaq-qec` imports, the harness adds NVIDIA's `nv-qldpc-decoder` as a third backend automatically. The adapter was written from NVIDIA's public docs and may need small kwarg changes for your installed version (see `src/qechybrid/cudaq_qec_adapter.py`)."),
    ("code", SETUP),
    ("code", "!pip -q install cudaq-qec"),
    ("code", "# restart runtime if the import below fails, then re-run\nfrom qechybrid import cudaq_qec_adapter\nprint(cudaq_qec_adapter.available())"),
    ("code", "!python scripts/run_benchmarks.py --profile full --device cuda --tag colab-qldpc --only qldpc"),
])

nb("03_ising_head_to_head.ipynb", [
    ("md", "# 03 - Head-to-head with NVIDIA's Ising pre-decoder\nRuns NVIDIA's real 3D-CNN pre-decoder + PyMatching, plain PyMatching, and **our** local pre-decoder on the *same* shots (NVIDIA's own circuit and noise model).\n\n**One-time access step:** the Ising weights are gated on Hugging Face. Open https://huggingface.co/nvidia/ising_decoder_surface_code_1_fast , accept the terms, create a *read* token, and add it in Colab (key icon, left bar) as a secret named `HF_TOKEN` with notebook access ON. The token stays in your Colab secrets; never paste it into a cell or commit it. The weights are under the NVIDIA Open Model License and are downloaded at run time only."),
    ("code", SETUP),
    ("code", 'import os, subprocess\nfrom google.colab import userdata\nos.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")  # read from Colab secrets, never printed\nsubprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/Ising-Decoding.git", "third_party/Ising-Decoding"], check=True)\n!pip -q install safetensors omegaconf hydra-core huggingface_hub beliefmatching'),
    ("code", "!python scripts/run_ising_bench.py --repo third_party/Ising-Decoding --download --device cuda --distances 9 13 --ps 0.003 0.005 --shots 20000 --tag colab-ising"),
    ("code", "!python scripts/make_report.py\n!zip -qr results_ising.zip results/colab-ising docs/index.html\nfrom google.colab import files; files.download('results_ising.zip')"),
    ("md", "Interpretation: compare LER (must match PyMatching), stage 1 vs stage 2 seconds, and total shots/s. Batch-1 latency is expected to favour plain PyMatching for NVIDIA's model too (see NVIDIA's own caveat in their cookbook)."),
])
RUN_ALL_SETUP = SETUP.replace(
    '# ---- edit this: your public GitHub repo ----\n',
    '# ---- settings ----\nPROFILE = "quick"   # "quick" (~5 min) first, then "full" (~20-40 min)\nRUN_ISING = True    # needs the Colab secret HF_TOKEN (see the markdown cell above); skipped if missing\nRUN_CUDAQ = False   # optional: pip-installs cudaq-qec, which can change the environment\n',
)

ISING_CELL = """import os, subprocess
if RUN_ISING:
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")   # never printed
    except Exception as exc:
        print("Skipping Ising head-to-head: no HF_TOKEN secret found (", type(exc).__name__, ")")
        RUN_ISING = False
if RUN_ISING:
    if not os.path.exists("third_party/Ising-Decoding"):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/Ising-Decoding.git", "third_party/Ising-Decoding"], check=True)
    subprocess.run("pip -q install safetensors omegaconf hydra-core huggingface_hub beliefmatching", shell=True)
    subprocess.run("python scripts/run_ising_bench.py --repo third_party/Ising-Decoding --download --device cuda --distances 9 13 --ps 0.003 0.005 --shots 20000 --tag colab-ising", shell=True)
    subprocess.run("python scripts/run_full_pipeline.py --repo third_party/Ising-Decoding --download --device cuda --distances 9 13 --ps 0.003 0.005 --shots 20000 --cpu-shots 20000 --tag colab-pipeline", shell=True)"""

CUDAQ_CELL = """import subprocess
if RUN_CUDAQ:
    subprocess.run("pip -q install cudaq-qec", shell=True)
    from qechybrid import cudaq_qec_adapter
    print("cudaq_qec:", cudaq_qec_adapter.available())
    subprocess.run("python scripts/run_benchmarks.py --profile quick --device cuda --tag colab-qldpc-cudaq --only qldpc", shell=True)
else:
    print("CUDA-Q QEC step skipped (RUN_CUDAQ = False)")"""

SUMMARY_CELL = """import glob, pandas as pd
for f in sorted(glob.glob("results/colab-*/*.csv")):
    df = pd.read_csv(f)
    print("\\n==", f, len(df), "rows")
    cols = [c for c in ["code", "d", "p", "decoder", "ler", "throughput_sps", "accept_rate"] if c in df.columns]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))"""

nb("04_run_everything_colab.ipynb", [
    ("md", "# 04 - Run everything (Runtime > Run all)\n"
           "Runs, in order: setup check, unit tests, GPU benchmarks (surface code + qLDPC), the NVIDIA Ising head-to-head, the full AI-pre-decoder + GPU-decoder pipeline vs CPU baselines, an optional CUDA-Q QEC step, then builds the website and downloads one zip.\n\n"
           "**Before you run:** Runtime > Change runtime type > **T4 GPU**. Settings are in the first code cell. Start with `PROFILE = \"quick\"`.\n\n"
           "**Ising step (optional):** accept the terms at https://huggingface.co/nvidia/ising_decoder_surface_code_1_fast , create a *read* token and add it as the Colab secret `HF_TOKEN` (key icon, notebook access ON). Without it that step is skipped and everything else still runs. Never paste the token into a cell."),
    ("code", RUN_ALL_SETUP),
    ("code", "!pip -q install pytest\n!pytest -q"),
    ("code", '!python scripts/run_benchmarks.py --profile $PROFILE --device cuda --tag colab-gpu'),
    ("code", ISING_CELL),
    ("code", CUDAQ_CELL),
    ("code", '!python scripts/make_report.py\n!zip -qr results_colab.zip results docs/index.html docs/figs'),
    ("code", SUMMARY_CELL),
    ("code", "from google.colab import files\nfiles.download('results_colab.zip')"),
])
print("notebooks written to", OUT)
