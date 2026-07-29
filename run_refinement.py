#!/usr/bin/env python3
"""Run the refinement experiments using the best configuration found."""
import torch
from diffusion_ik import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {device}")
log(f"GPU: {torch.cuda.get_device_name(0)}")

# Best config from research: ResMLP large + eps + cosine T=200 + CFG
best_arch = dict(arch="resmlp", hidden=1024, layers=12)
best_adv = dict(cfg_dropout=0.1, guidance_scale=1.5)

R = {}

# Re-run the CFG experiment for baseline
log("\n" + "#" * 60)
log("  Baseline: CFG (no refinement)")
log("#" * 60)

R["baseline"] = run_experiment(base(
    tag="v14_cfg_baseline", pred_type="eps", schedule="cosine", T=200,
    **best_arch, **best_adv,
    notes="CFG baseline (no refinement)"), device)

# Refinement with 50 steps
log("\n" + "#" * 60)
log("  Refinement: 50 steps")
log("#" * 60)

R["refine50"] = run_experiment(base(
    tag="v15_refine50", pred_type="eps", schedule="cosine", T=200,
    refine_steps=50, refine_lr=0.01,
    **best_arch, **best_adv,
    notes="CFG + refinement 50 steps"), device)

# Refinement with 200 steps
log("\n" + "#" * 60)
log("  Refinement: 200 steps")
log("#" * 60)

R["refine200"] = run_experiment(base(
    tag="v16_refine200", pred_type="eps", schedule="cosine", T=200,
    refine_steps=200, refine_lr=0.005,
    **best_arch, **best_adv,
    notes="CFG + refinement 200 steps"), device)

# Refinement with 500 steps (maximum precision)
log("\n" + "#" * 60)
log("  Refinement: 500 steps (max precision)")
log("#" * 60)

R["refine500"] = run_experiment(base(
    tag="v17_refine500", pred_type="eps", schedule="cosine", T=200,
    refine_steps=500, refine_lr=0.003,
    **best_arch, **best_adv,
    notes="CFG + refinement 500 steps"), device)

# Summary
log("\n" + "=" * 60)
log("  REFINEMENT RESULTS")
log("=" * 60)
for k in sorted(R):
    r = R[k]
    log(f"  {r['tag']:25s} SR={r['success_rate']:6.2%}  pos={r['pos_mm']:7.1f}mm  "
        f"time={r['time']:5.0f}s")
bst = pick_best(R, R)
log(f"\n  BEST: {R[bst]['tag']} SR={R[bst]['success_rate']:.2%} pos={R[bst]['pos_mm']:.1f}mm")
log("=" * 60)
