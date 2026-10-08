"""Head-to-head with NVIDIA's Ising 3D-CNN pre-decoder (https://github.com/NVIDIA/Ising-Decoding, Apache-2.0).

We do NOT vendor NVIDIA's code. Clone it next to this repo (the Colab notebook does) and pass its path.
The wiring below follows NVIDIA's own cookbook (`cookbook/predecoder.ipynb`):

    Stim syndromes (their MemoryCircuit + 25-parameter noise model, boundary detectors)
      -> PreDecoderMemoryEvalModule  (3D CNN, returns [partial logical flip | residual detectors])
      -> PyMatching on the residual   -> XOR of both flips

and compares it on the SAME shots against (a) plain PyMatching and (b) our own local pre-decoder.

Model weights (NVIDIA Open Model License) are gated on Hugging Face: you need to accept the terms there
and provide your own token (env var HF_TOKEN or a cached `huggingface-cli login`). Tokens are never
stored or printed here. With `weights=None` the network is randomly initialised: the plumbing and timing
can be checked, but the logical error rate is meaningless and rows are labelled accordingly.
"""
from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import numpy as np

from .timing import now, percentiles_us, wilson

# Candidate repo ids per model (the current public names first, NVIDIA's older cookbook names as fallback).
HF_REPOS = {
    1: ["nvidia/Ising-Decoder-SurfaceCode-1-Fast", "nvidia/ising_decoder_surface_code_1_fast"],
    2: ["nvidia/Ising-Decoder-SurfaceCode-1-Accurate", "nvidia/ising_decoder_surface_code_1_accurate"],
}


def download_weights(model_id: int = 1) -> str:
    """Download the gated weights with the caller's own Hugging Face credentials.

    Looks the .safetensors file up in the repo (so a version bump in the file name does not break us)
    and prefers the fp16 checkpoint. The token is read from HF_TOKEN or a cached login and never printed.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    token = os.environ.get("HF_TOKEN") or True
    last = None
    for repo in HF_REPOS[model_id]:
        try:
            files = [f for f in list_repo_files(repo, token=token) if f.endswith(".safetensors")]
            if not files:
                raise FileNotFoundError(f"no .safetensors file in {repo}")
            fname = next((f for f in files if "fp16" in f), files[0])
            return hf_hub_download(repo_id=repo, filename=fname, token=token)
        except Exception as exc:  # try the next candidate name
            last = exc
    raise RuntimeError(
        f"could not download Ising weights for model {model_id} (last error: {type(last).__name__}). "
        "Check that the HF_TOKEN secret is set and that your account has been granted access to the model."
    ) from last


def _import_nvidia(repo: str):
    code = os.path.join(os.path.abspath(repo), "code")
    if not os.path.isdir(code):
        raise FileNotFoundError(f"{code} not found: clone https://github.com/NVIDIA/Ising-Decoding first")
    if code not in sys.path:
        sys.path.insert(0, code)
    from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps
    from model.factory import ModelFactory
    from model.registry import get_model_spec
    from qec.noise_model import NoiseModel
    from qec.surface_code.memory_circuit import MemoryCircuit

    return PreDecoderMemoryEvalModule, _build_stab_maps, ModelFactory, get_model_spec, NoiseModel, MemoryCircuit


def build_context(repo, distance=9, n_rounds=None, p=0.005, shots=5000, basis="X", code_rotation="XV",
                  model_id=1, weights=None, device="cpu", seed=0):
    import pymatching
    import torch

    Pipe, build_maps, Factory, get_spec, Noise, Circuit = _import_nvidia(repo)
    n_rounds = n_rounds or distance
    noise = Noise.from_single_p(p)
    pmax = noise.get_max_probability()
    circ = Circuit(distance=distance, n_rounds=n_rounds, basis=basis, code_rotation=code_rotation,
                   idle_error=pmax, sqgate_error=pmax, tqgate_error=pmax, spam_error=(2 / 3) * pmax,
                   noise_model=noise, add_boundary_detectors=True)
    circ.set_error_rates()
    sc = circ.stim_circuit
    meas = sc.compile_sampler(seed=seed).sample(shots)
    do = sc.compile_m2d_converter().convert(measurements=meas, append_observables=True)
    nobs = sc.num_observables
    det, obs = do[:, :-nobs].astype(np.uint8), do[:, -nobs:].astype(np.uint8)
    dem = sc.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    spec = get_spec(model_id)
    mcfg = SimpleNamespace(code="surface", distance=distance, n_rounds=n_rounds, model=SimpleNamespace(
        version="predecoder_memory_v1", num_filters=list(spec.num_filters), kernel_size=list(spec.kernel_size),
        dropout_p=0.0, activation="gelu", input_channels=4, out_channels=4))
    model = Factory.create_model(mcfg)
    if weights:
        from safetensors.torch import load_file

        sd = load_file(weights, device="cpu")
        model.load_state_dict({(k[7:] if k.startswith("module.") else k): v.float() for k, v in sd.items()})
    model = model.to(device).eval()
    ecfg = SimpleNamespace(distance=distance, enable_fp16=False, data=SimpleNamespace(code_rotation=code_rotation),
                           test=SimpleNamespace(meas_basis_test=basis, th_data=0.0, th_syn=0.0, sampling_mode="threshold",
                                                temperature=1.0, temperature_data=None, temperature_syn=None, n_rounds=n_rounds))
    pipe = Pipe(model, ecfg, build_maps(distance, code_rotation), torch.device(device)).to(device).eval()
    return SimpleNamespace(pipe=pipe, det=det, obs=obs[:, 0], matcher=matcher, circuit=sc, device=device, d=distance, rounds=n_rounds,
                           p=p, trained=bool(weights), model_id=model_id, shots=shots)


CHUNK = 2048  # fixed batch shape: one compile, bounded activation memory


def ising_forward(ctx, det_u8, chunk: int = CHUNK):
    """Run NVIDIA's pipeline over `det_u8` (numpy uint8 [n, N]) in fixed-size chunks.

    Every call uses exactly `chunk` rows (the last chunk is zero-padded and sliced back), so torch.compile
    sees a single shape and a 20k-shot batch cannot exhaust GPU memory in one pass.
    Returns the pipeline output [n, 1+N] as a tensor on the device; time it with a device sync around the call.
    """
    import torch

    n = len(det_u8)
    chunk = min(chunk, n)  # small batches use their own (single) shape; no padding
    outs = []
    with torch.no_grad():
        for i in range(0, n, chunk):
            x = torch.from_numpy(np.ascontiguousarray(det_u8[i : i + chunk])).to(torch.uint8)
            m = x.shape[0]
            if m < chunk:
                x = torch.cat([x, torch.zeros(chunk - m, x.shape[1], dtype=torch.uint8)])
            outs.append(ctx.pipe(x.to(ctx.device))[:m])
    return torch.cat(outs)


def ising_warmup(ctx, det_u8, chunk: int = CHUNK):
    chunk = min(chunk, len(det_u8))
    ising_forward(ctx, det_u8[:chunk], chunk)
    ising_forward(ctx, det_u8[:chunk], chunk)


def run_comparison(ctx, lat_shots: int = 200, log=print) -> list[dict]:
    """Rows (same schema as the surface benchmark) for PyMatching, NVIDIA Ising + PyMatching, our local pre-decoder."""
    import torch

    from .local_predecoder import LocalPreDecoder
    from .pipeline import HybridDecoder

    dev, det, obs, m = ctx.device, ctx.det, ctx.obs, ctx.matcher

    class _Matcher:  # HybridDecoder expects decode_batch / decode_one
        decode_batch = staticmethod(lambda d: np.asarray(m.decode_batch(d), dtype=np.uint8).reshape(len(d), -1))
        decode_one = staticmethod(lambda x: np.asarray(m.decode(x), dtype=np.uint8))

    n = len(det)
    tag = "" if ctx.trained else " [RANDOM WEIGHTS: LER not meaningful]"
    rows = []

    def row(name, errs, thr, lat, accept, kept, t1, t2, extra=""):
        ler, lo, hi = wilson(errs, n)
        return dict(code="surface-nvidia-circuit", d=ctx.d, rounds=ctx.rounds, p=ctx.p, decoder=name + (tag if "Ising" in name else ""), device=dev, shots=n,
                    errors=errs, ler=ler, ler_lo=lo, ler_hi=hi, throughput_sps=thr, lat_p50_us=lat.get("p50", np.nan),
                    lat_p95_us=lat.get("p95", np.nan), lat_p99_us=lat.get("p99", np.nan), accept_rate=accept,
                    syndrome_weight_kept=kept, t_stage1_s=t1, t_global_s=t2, extra=extra)

    def lat_of(fn):
        try:
            items = [det[i] for i in range(min(lat_shots, n))]
            for x in items[:10]:
                fn(x)
            ts = []
            for x in items:
                t0 = now(dev)
                fn(x)
                ts.append(now(dev) - t0)
            return percentiles_us(ts)
        except Exception as exc:  # e.g. recompilation limits at batch size 1
            log(f"  latency skipped: {type(exc).__name__}")
            return {}

    # 1) PyMatching only
    m.decode_batch(det[:500])
    t0 = time.perf_counter()
    pm = np.asarray(m.decode_batch(det), dtype=np.uint8).reshape(n, -1)[:, 0]
    t_pm = time.perf_counter() - t0
    rows.append(row("pymatching (CPU)", int((pm != obs).sum()), n / t_pm, lat_of(lambda x: m.decode(x)), 0.0, 1.0, 0.0, t_pm))

    # 2) NVIDIA Ising + PyMatching
    ising_warmup(ctx, det)  # compile once for the fixed chunk shape (outside the timed region)
    t0 = now(dev)
    out = ising_forward(ctx, det)
    t_pd = now(dev) - t0
    flip = out[:, 0].cpu().numpy().astype(np.uint8)
    res = out[:, 1:].cpu().numpy().astype(np.uint8)
    t0 = time.perf_counter()
    pmr = np.asarray(m.decode_batch(res), dtype=np.uint8).reshape(n, -1)[:, 0]
    t_pr = time.perf_counter() - t0
    pred = flip ^ pmr

    def ising_one(x):
        with torch.no_grad():
            o = ctx.pipe(torch.from_numpy(x[None]).to(torch.uint8).to(dev))
        r = o[0, 1:].cpu().numpy().astype(np.uint8)
        return int(o[0, 0].item()) ^ int(m.decode(r)[0])

    rows.append(row("NVIDIA Ising + pymatching", int((pred != obs).sum()), n / (t_pd + t_pr), lat_of(ising_one),
                    float((res.sum(1) == 0).mean()), float(res.sum()) / max(1.0, float(det.sum())), t_pd, t_pr,
                    extra=f"model_id={ctx.model_id}"))

    # 3) our local pre-decoder + PyMatching, same shots
    hd = HybridDecoder(_Matcher, mode="local", device=dev, local=LocalPreDecoder(m, device=dev))
    hd.decode_batch(det[:500])
    p3 = hd.decode_batch(det)[:, 0]
    s = hd.stats
    rows.append(row("ours: local pre-decoder + pymatching", int((p3 != obs).sum()), n / s["t_total"], lat_of(lambda x: hd.decode_one(x)),
                    s["accept_rate"], s["syndrome_weight_kept"], s["t_gate"], s["t_match"]))
    for r in rows:
        log(f"  d={ctx.d} p={ctx.p} {r['decoder']:55s} LER={r['ler']:.2e} {r['throughput_sps']:,.0f} shots/s")
    return rows


def run_full_pipeline(ctx, cpu_shots: int = 1000, max_iter: int = 30, log=print) -> list[dict]:
    """End-to-end comparison on identical shots:

      CPU baselines : PyMatching; CPU BP+OSD-0 (ldpc)                  (BP+OSD-0 on the first `cpu_shots` shots only: it is slow)
      AI pre-decoder: NVIDIA Ising + PyMatching; Ising + CPU BP+OSD-0
      FULL PIPELINE : NVIDIA Ising (GPU) -> GPU BP (+ CPU OSD-0 fallback for the shots BP cannot resolve)

    Every row reports logical errors and total wall-clock time. The full-pipeline
    timing includes Ising inference, GPU-side postprocessing, GPU BP, CPU fallback,
    and the host transfer of the partial logical bit; GPU timers are synchronized.
    """
    import torch

    from .data import dem_matrices
    from .decoders import CpuBpOsd, DemBpOsd

    dev, det, obs, m = ctx.device, ctx.det, ctx.obs, ctx.matcher
    n = len(det)
    k = min(n, cpu_shots)
    tag = "" if ctx.trained else " [RANDOM WEIGHTS: LER not meaningful]"
    rows = []

    def add(name, errs, shots, t1, t2, extra="", kept=np.nan):
        ler, lo, hi = wilson(errs, shots)
        total = t1 + t2
        rows.append(dict(code="surface-nvidia-circuit", d=ctx.d, rounds=ctx.rounds, p=ctx.p, decoder=name + (tag if "Ising" in name else ""),
                         device=dev, shots=shots, errors=errs, ler=ler, ler_lo=lo, ler_hi=hi, throughput_sps=shots / total,
                         total_time_s=total, us_per_shot=1e6 * total / shots, t_stage1_s=t1, t_global_s=t2,
                         syndrome_weight_kept=kept, extra=extra))
        log(f"  d={ctx.d} p={ctx.p} {rows[-1]['decoder']:62s} errs={errs}/{shots} {1e6 * total / shots:9.1f} us/shot")

    # --- CPU baseline 1: PyMatching ---
    m.decode_batch(det[:500])
    t0 = time.perf_counter()
    pm = np.asarray(m.decode_batch(det), dtype=np.uint8).reshape(n, -1)[:, 0]
    add("CPU: pymatching", int((pm != obs).sum()), n, 0.0, time.perf_counter() - t0)

    dm = dem_matrices(ctx.circuit)
    cpu_osd = CpuBpOsd(dm.H, dm.priors, max_iter=max_iter, osd_order=0, osd_method="osd_0")
    gpu_dec = DemBpOsd(dm, device=dev, max_iter=max_iter)

    # --- CPU baseline 2: BP+OSD-0 on raw syndromes (subset) ---
    try:
        t0 = time.perf_counter()
        e = cpu_osd.decode_batch(det[:k])
        t = time.perf_counter() - t0
        add("CPU: BP+OSD-0 (ldpc)", int((gpu_dec.obs_from_errors(e)[:, 0] != obs[:k]).sum()), k, 0.0, t, extra=f"first {k} shots only")
    except Exception as exc:
        log(f"  CPU BP+OSD-0 failed: {type(exc).__name__}: {exc}")

    # --- Ising stage 1 (GPU), timed once on the full batch ---
    # Compile/warm-up is outside the measurement. GPU postprocessing that converts the
    # model output into residual/flip tensors is included in stage 1.
    ising_warmup(ctx, det)
    t0 = now(dev)
    out = ising_forward(ctx, det)
    flip_t = out[:, 0].round().to(torch.uint8)
    res_t = out[:, 1:].round().to(torch.uint8)
    t_pd = now(dev) - t0

    # --- FULL PIPELINE: Ising (GPU) -> GPU BP (+ OSD-0 fallback) ---
    # Keep the residual on the GPU. The partial logical bit is transferred once,
    # after GPU BP/CPU fallback, and that transfer is included in stage 2.
    try:
        gpu_dec.decode_batch(res_t[:256])  # warm up kernels
        t0 = now(dev)
        g = gpu_dec.decode_batch(res_t)
        flip_full = flip_t.cpu().numpy()
        t2 = now(dev) - t0
        kept = float(res_t.sum().item()) / max(1.0, float(det.sum()))
        add("FULL: NVIDIA Ising (GPU) -> GPU BP (+OSD-0 fallback)", int(((flip_full ^ g[:, 0]) != obs).sum()), n, t_pd, t2, kept=kept,
            extra=f"BP fallback to CPU OSD-0 on {100 * gpu_dec.last['fallback_frac']:.1f}% of shots")
    except Exception as exc:
        log(f"  FULL pipeline failed: {type(exc).__name__}: {exc}")

    # Host copies below are diagnostics for the CPU comparison rows and are deliberately
    # outside the full-pipeline timing above.
    flip = flip_full if "flip_full" in locals() else flip_t.cpu().numpy()
    res = res_t.cpu().numpy()
    kept = float(res.sum()) / max(1.0, float(det.sum()))

    # --- AI + PyMatching (CPU global decoder) ---
    t0 = time.perf_counter()
    pmr = np.asarray(m.decode_batch(res), dtype=np.uint8).reshape(n, -1)[:, 0]
    add("NVIDIA Ising (GPU) + pymatching (CPU)", int(((flip ^ pmr) != obs).sum()), n, t_pd, time.perf_counter() - t0, kept=kept)

    # --- AI + CPU BP+OSD-0 (subset) ---
    try:
        t0 = time.perf_counter()
        e = cpu_osd.decode_batch(res[:k])
        t = time.perf_counter() - t0
        pred = flip[:k] ^ gpu_dec.obs_from_errors(e)[:, 0]
        add("NVIDIA Ising (GPU) + BP+OSD-0 (CPU)", int((pred != obs[:k]).sum()), k, t_pd * k / n, t, kept=kept, extra=f"BP+OSD-0 stage on first {k} shots; stage 1 scaled to {k}")
    except Exception as exc:
        log(f"  Ising + CPU BP+OSD-0 failed: {type(exc).__name__}: {exc}")
    return rows
