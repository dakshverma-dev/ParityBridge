import numpy as np
import torch

from qechybrid.ising_adapter import (
    build_context,
    download_weights,
    ising_forward,
    ising_warmup,
)
from qechybrid.data import dem_matrices
from qechybrid.decoders import CpuBpOsd, DemBpOsd

REPO = r".\third_party\Ising-Decoding"
SHOTS = 20000
DISTANCE = 9
P = 0.005

print("Loading trained Ising weights...")
weights = download_weights(model_id=1)

print("Building context...")
ctx = build_context(
    REPO,
    distance=DISTANCE,
    p=P,
    shots=SHOTS,
    device="cuda",
    seed=0,
    weights=weights,
    model_id=1,
)

det = ctx.det
obs = ctx.obs.astype(np.uint8)

print(f"Shots: {len(det)}")
print(f"Detector shape: {det.shape}")

print("Building decoders...")
dm = dem_matrices(ctx.circuit)

cpu = CpuBpOsd(
    dm.H,
    dm.priors,
    max_iter=30,
    osd_order=0,
    osd_method="osd_0",
)

hybrid = DemBpOsd(
    dm,
    device="cuda",
    max_iter=30,
)

print("Running trained Ising...")
ising_warmup(ctx, det)
out = ising_forward(ctx, det)

flip = (
    out[:, 0]
    .round()
    .cpu()
    .numpy()
    .astype(np.uint8)
)

res = (
    out[:, 1:]
    .round()
    .cpu()
    .numpy()
    .astype(np.uint8)
)

print(f"Residual shape: {res.shape}")

print("Running Ising + CPU BP+OSD-0...")
cpu_errors = cpu.decode_batch(res)
cpu_residual_logical = hybrid.obs_from_errors(cpu_errors)[:, 0]
cpu_logical = flip ^ cpu_residual_logical

print("Running Ising + GPU BP + CPU fallback...")
res_gpu = torch.as_tensor(res, dtype=torch.uint8, device="cuda")
gpu_residual_logical = hybrid.decode_batch(res_gpu)[:, 0]
gpu_logical = flip ^ gpu_residual_logical

cpu_correct = cpu_logical == obs
gpu_correct = gpu_logical == obs
agree = cpu_logical == gpu_logical

print()
print("=== TRAINED MODEL PAIRED AUDIT ===")
print(f"CPU Ising+BP+OSD correct:  {int(cpu_correct.sum())}/{SHOTS}")
print(f"GPU hybrid correct:        {int(gpu_correct.sum())}/{SHOTS}")
print(f"Both correct:              {int((cpu_correct & gpu_correct).sum())}")
print(f"Both wrong:                {int((~cpu_correct & ~gpu_correct).sum())}")
print(f"CPU only correct:          {int((cpu_correct & ~gpu_correct).sum())}")
print(f"GPU only correct:          {int((~cpu_correct & gpu_correct).sum())}")
print(f"Decoder agreement:         {int(agree.sum())}/{SHOTS}")
print(f"Decoder disagreement:      {int((~agree).sum())}")
print(f"CPU errors:                {int((~cpu_correct).sum())}")
print(f"GPU hybrid errors:         {int((~gpu_correct).sum())}")
print(f"CPU fallback to OSD-0:              {hybrid.last['n_fallback']}/{SHOTS}")
print(f"Fallback fraction:         {100 * hybrid.last['fallback_frac']:.1f}%")
