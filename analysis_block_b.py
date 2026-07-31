#!/usr/bin/env python3
"""
analysis_block_b.py — Diversity & refinement analyses for the DiffusionIK paper revision.

Implements the analyses requested by Reviewer 1 (comment 2) on a single trained
checkpoint and a single fixed (seeded) set of test poses:

  B1  Per-candidate success rate (fraction of the M generated candidates that meet
      the 1 mm / 1 deg thresholds) after 0, 50, 200 and 500 refinement steps.
  B2  Clustering of refined configurations in joint space (single-linkage with a
      distance threshold) -> number of distinct solution modes per pose.
  B3  Convergence curves: mean and median task-space residual (position and
      orientation) vs refinement step, for each refinement regime.
  B4  Because every number is produced from ONE checkpoint and ONE fixed test set,
      the run also yields a self-consistent replacement for the pre-refinement SR
      values that differ between Tables 4-7 of the original manuscript.

The refinement regimes mirror the paper: 50 steps @ lr 0.01, 200 @ 0.005,
500 @ 0.003, each starting from the same initial diffusion candidates.

Note: to stay faithful to the code that produced the published results, the
refinement update does NOT clip to joint limits at every step (diffusion_ik._refine
has no clamp). The script instead *measures* the fraction of refined candidates
that leave the admissible box and reports it, so the paper text can be aligned
with the implementation.

Usage:
    uv run python analysis_block_b.py --ckpt best_v14_cfg_baseline.pt
    # smoke test without a checkpoint (random weights, tiny sizes):
    uv run python analysis_block_b.py --random-init --quick

Outputs (in --out, default results_block_b/):
    results.json      all metrics
    arrays.npz        test poses, initial/refined joints, per-step curves
    tables.md         ready-to-paste markdown tables
    fig_convergence_grid.(png|pdf)   3 regimes x (position, orientation)
    fig_convergence_200.(png|pdf)    single-regime 2-panel version (200 steps)
    fig_per_candidate_sr.(png|pdf)   B1 summary
    fig_modes_hist.(png|pdf)         B2 histograms for the tau sweep
"""

import argparse
import hashlib
import json
import math
import os
import time
from math import pi

import torch
import torch.nn.functional as F

from fk import FK
from common import JOINTS_LOWER_LIMIT, JOINTS_UPPER_LIMIT
from eval_ik import geodesic_distance
from diffusion_ik import (
    ResMLPDenoiser, NoiseScheduler, JointNormalizer, sample_loop,
)

RAD2DEG = 180.0 / pi
M2MM = 1000.0


def log(msg):
    print(msg, flush=True)


# ─────────────────────────────────────────────────────────────
# Test set
# ─────────────────────────────────────────────────────────────

def make_test_set(n_targets, seed, device):
    """Fixed, seeded set of reachable target poses (sampled in joint space + FK)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    lo = torch.tensor(JOINTS_LOWER_LIMIT)
    hi = torch.tensor(JOINTS_UPPER_LIMIT)
    q = torch.rand(n_targets, 7, generator=g) * (hi - lo) + lo
    q = q.to(device)
    fk = FK(device)
    with torch.no_grad():
        R, t = fk(q)
    return q, R, t


# ─────────────────────────────────────────────────────────────
# Errors / metrics
# ─────────────────────────────────────────────────────────────

def task_errors(fk, joints, R_tgt, t_tgt):
    """Position (m) and orientation (rad) errors. joints: (B,7); targets broadcast."""
    R_p, t_p = fk(joints)
    pos = (t_p - t_tgt).squeeze(-1).norm(dim=-1)
    ori = geodesic_distance(R_p, R_tgt)
    return pos, ori


def summarize_errors(pos, ori, pos_thr, ori_thr, n_targets, m_samples,
                     joints=None, lo=None, hi=None):
    """All headline metrics for a (n_targets*m_samples,) flat error vector."""
    succ = (pos < pos_thr) & (ori < ori_thr)
    succ_pose = succ.view(n_targets, m_samples)
    out = {
        "sr_any_pct": succ_pose.any(dim=1).float().mean().item() * 100,
        "sr_per_candidate_pct": succ.float().mean().item() * 100,
        "per_pose_success_frac": {
            "mean_pct": succ_pose.float().mean(dim=1).mean().item() * 100,
            "median_pct": succ_pose.float().mean(dim=1).median().item() * 100,
            "p10_pct": succ_pose.float().mean(dim=1).quantile(0.10).item() * 100,
            "p90_pct": succ_pose.float().mean(dim=1).quantile(0.90).item() * 100,
        },
        "pos_mm": {
            "mean": pos.mean().item() * M2MM,
            "median": pos.median().item() * M2MM,
            "p95": pos.quantile(0.95).item() * M2MM,
        },
        "ori_deg": {
            "mean": ori.mean().item() * RAD2DEG,
            "median": ori.median().item() * RAD2DEG,
            "p95": ori.quantile(0.95).item() * RAD2DEG,
        },
    }
    if joints is not None:
        viol = ((joints < lo) | (joints > hi)).any(dim=-1)
        out["out_of_limits_pct"] = viol.float().mean().item() * 100
    return out


def diversity_metrics(joints, succ, n_targets, m_samples):
    """Paper metric: mean pairwise joint-space distance among successful samples,
    averaged over poses with >= 2 successes. Also over all candidates."""
    J = joints.view(n_targets, m_samples, 7)
    S = succ.view(n_targets, m_samples)
    div_succ, div_all = [], []
    for i in range(n_targets):
        Ji = J[i]
        d_all = torch.pdist(Ji)
        if d_all.numel():
            div_all.append(d_all.mean().item())
        Js = Ji[S[i]]
        if Js.shape[0] >= 2:
            div_succ.append(torch.pdist(Js).mean().item())
    return {
        "diversity_successful_rad": sum(div_succ) / len(div_succ) if div_succ else 0.0,
        "diversity_all_rad": sum(div_all) / len(div_all) if div_all else 0.0,
        "poses_with_ge2_successes": len(div_succ),
    }


# ─────────────────────────────────────────────────────────────
# B2: single-linkage clustering with a joint-space threshold
# ─────────────────────────────────────────────────────────────

def count_modes(J, tau):
    """Number of connected components with edges d(i,j) < tau. J: (K,7)."""
    K = J.shape[0]
    if K == 0:
        return 0
    D = torch.cdist(J, J)
    parent = list(range(K))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    idx = (D < tau).nonzero()
    for i, j in idx.tolist():
        if i < j:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    return len({find(i) for i in range(K)})


def clustering_analysis(joints, succ, n_targets, m_samples, taus):
    J = joints.view(n_targets, m_samples, 7)
    S = succ.view(n_targets, m_samples)
    res = {}
    for tau in taus:
        counts = []
        for i in range(n_targets):
            Js = J[i][S[i]]
            if Js.shape[0] == 0:
                continue
            counts.append(count_modes(Js.cpu(), tau))
        if not counts:
            res[f"{tau:g}"] = {"poses_evaluated": 0, "mean_modes": 0.0,
                               "median_modes": 0.0, "min_modes": 0, "max_modes": 0,
                               "histogram": {}}
            continue
        c = torch.tensor(counts, dtype=torch.float32)
        hist = torch.bincount(torch.tensor(counts, dtype=torch.long), minlength=1)
        res[f"{tau:g}"] = {
            "poses_evaluated": len(counts),
            "mean_modes": c.mean().item(),
            "median_modes": c.median().item(),
            "min_modes": int(c.min().item()),
            "max_modes": int(c.max().item()),
            "histogram": {str(k): int(v) for k, v in enumerate(hist.tolist()) if v > 0},
        }
    return res


# ─────────────────────────────────────────────────────────────
# Sampling + refinement with per-step curve logging
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_candidates(model, ns, norm, cfg, R_t, t_t, m_samples, chunk, device):
    """DDIM+CFG candidates for every target. Returns joints (N*M, 7) and targets
    expanded to candidate granularity."""
    n = R_t.shape[0]
    all_joints = []
    t0 = time.time()
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        b = e - s
        cond = torch.cat([R_t[s:e].reshape(b, 9), t_t[s:e].reshape(b, 3)], dim=1)
        cond = cond.repeat_interleave(m_samples, dim=0)
        x = sample_loop(model, ns, cond, device, cfg)
        all_joints.append(norm.denormalize(x.clamp(-1, 1)))
        done = e
        el = time.time() - t0
        log(f"  sampling: {done}/{n} targets, {el:.0f}s elapsed, "
            f"ETA {el / done * (n - done):.0f}s")
    joints = torch.cat(all_joints, 0)
    R_exp = R_t.repeat_interleave(m_samples, dim=0)
    t_exp = t_t.repeat_interleave(m_samples, dim=0)
    return joints, R_exp, t_exp, time.time() - t0


def refine_with_curves(joints0, R_exp, t_exp, fk, steps, lr, snap_steps, device):
    """Adam refinement through differentiable FK (mirrors diffusion_ik._refine,
    including the absence of per-step clipping). Logs mean/median errors at every
    step and snapshots joints at snap_steps."""
    j = joints0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([j], lr=lr)
    curve = {"step": [], "pos_mean": [], "pos_median": [], "ori_mean": [], "ori_median": []}
    snaps = {}
    t0 = time.time()
    for k in range(1, steps + 1):
        R_p, t_p = fk(j)
        loss = F.mse_loss(t_p, t_exp) + F.mse_loss(R_p, R_exp)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            pos = (t_p - t_exp).squeeze(-1).norm(dim=-1)
            ori = geodesic_distance(R_p, R_exp)
            curve["step"].append(k - 1)  # errors BEFORE this update (state at step k-1)
            curve["pos_mean"].append(pos.mean().item() * M2MM)
            curve["pos_median"].append(pos.median().item() * M2MM)
            curve["ori_mean"].append(ori.mean().item() * RAD2DEG)
            curve["ori_median"].append(ori.median().item() * RAD2DEG)
        if k in snap_steps:
            snaps[k] = j.detach().clone()
        if k % max(1, steps // 10) == 0:
            log(f"    refine lr={lr}: step {k}/{steps}, "
                f"pos_med={curve['pos_median'][-1]:.4f}mm")
    # final state errors (after the last update)
    with torch.no_grad():
        pos, ori = task_errors(fk, j.detach(), R_exp, t_exp)
        curve["step"].append(steps)
        curve["pos_mean"].append(pos.mean().item() * M2MM)
        curve["pos_median"].append(pos.median().item() * M2MM)
        curve["ori_mean"].append(ori.mean().item() * RAD2DEG)
        curve["ori_median"].append(ori.median().item() * RAD2DEG)
    snaps[steps] = j.detach().clone()
    return snaps, curve, time.time() - t0


# ─────────────────────────────────────────────────────────────
# Figures (journal style: white bg, recessive grid, no top/right spines)
# ─────────────────────────────────────────────────────────────

BLUE = "#2a78d6"
ORANGE = "#eb6834"

def _style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="both", color="#e6e6e6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def make_figures(res, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "legend.fontsize": 8, "figure.dpi": 120, "savefig.dpi": 300,
    })

    regimes = list(res["convergence"].keys())  # e.g. ["50", "200", "500"]

    # ── B3 grid: rows = (position, orientation), cols = regimes ──
    fig, axes = plt.subplots(2, len(regimes), figsize=(2.6 * len(regimes), 4.4),
                             sharex="col")
    if len(regimes) == 1:
        axes = axes.reshape(2, 1)
    for c, reg in enumerate(regimes):
        cv = res["convergence"][reg]
        lr = res["regimes"][reg]["refine_lr"]
        ax = axes[0, c]
        ax.plot(cv["step"], cv["pos_mean"], color=BLUE, lw=1.6, label="Mean")
        ax.plot(cv["step"], cv["pos_median"], color=BLUE, lw=1.4, ls="--", label="Median")
        ax.axhline(1.0, color="#999999", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_title(f"{reg} steps, lr={lr:g}")
        if c == 0:
            ax.set_ylabel("Position residual, mm")
            ax.legend(frameon=False)
        _style_ax(ax)
        ax = axes[1, c]
        ax.plot(cv["step"], cv["ori_mean"], color=BLUE, lw=1.6)
        ax.plot(cv["step"], cv["ori_median"], color=BLUE, lw=1.4, ls="--")
        ax.axhline(1.0, color="#999999", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel("Refinement step")
        if c == 0:
            ax.set_ylabel("Orientation residual, deg")
        _style_ax(ax)
    fig.suptitle("Task-space residual vs refinement step (dotted line: success threshold)",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig_convergence_grid.{ext}"))
    plt.close(fig)

    # ── single-regime 2-panel version (paper's default 200-step regime) ──
    reg = "200" if "200" in res["convergence"] else regimes[-1]
    cv = res["convergence"][reg]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, key, ylab in ((axes[0], "pos", "Position residual, mm"),
                          (axes[1], "ori", "Orientation residual, deg")):
        ax.plot(cv["step"], cv[f"{key}_mean"], color=BLUE, lw=1.6, label="Mean")
        ax.plot(cv["step"], cv[f"{key}_median"], color=BLUE, lw=1.4, ls="--", label="Median")
        ax.axhline(1.0, color="#999999", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel("Refinement step")
        ax.set_ylabel(ylab)
        _style_ax(ax)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig_convergence_{reg}.{ext}"))
    plt.close(fig)

    # ── B1: per-candidate and any-candidate SR vs refinement steps ──
    xs = [0] + [int(r) for r in regimes]
    per_cand = [res["pre_refinement"]["sr_per_candidate_pct"]] + \
               [res["regimes"][r]["sr_per_candidate_pct"] for r in regimes]
    any_cand = [res["pre_refinement"]["sr_any_pct"]] + \
               [res["regimes"][r]["sr_any_pct"] for r in regimes]
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    ax.plot(xs, any_cand, color=BLUE, lw=1.6, marker="o", ms=4,
            label="Any-candidate SR (per pose)")
    ax.plot(xs, per_cand, color=ORANGE, lw=1.6, marker="s", ms=4,
            label="Per-candidate SR")
    for x, y in zip(xs, per_cand):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, -11),
                    ha="center", fontsize=7.5, color="#555555")
    for x, y in zip(xs, any_cand):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7.5, color="#555555")
    ax.set_xlabel("Refinement steps")
    ax.set_ylabel("Success rate, %")
    ax.set_ylim(-4, 108)
    ax.legend(frameon=False, loc="center right")
    _style_ax(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig_per_candidate_sr.{ext}"))
    plt.close(fig)

    # ── B2: histogram of modes per pose for the tau sweep (final regime) ──
    reg = regimes[-1]
    taus = list(res["clustering"][reg].keys())
    fig, axes = plt.subplots(1, len(taus), figsize=(2.4 * len(taus), 2.6), sharey=True)
    if len(taus) == 1:
        axes = [axes]
    for ax, tau in zip(axes, taus):
        cl = res["clustering"][reg][tau]
        ks = sorted(int(k) for k in cl["histogram"])
        vs = [cl["histogram"][str(k)] for k in ks]
        ax.bar(ks, vs, color=BLUE, width=0.8, zorder=2)
        ax.set_title(f"τ = {tau} rad  (mean {cl['mean_modes']:.1f})")
        ax.set_xlabel("Distinct modes per pose")
        _style_ax(ax)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Number of poses")
    fig.suptitle(f"Solution modes among successful candidates after {reg} refinement steps",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig_modes_hist.{ext}"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Markdown tables
# ─────────────────────────────────────────────────────────────

def make_tables(res, outdir):
    lines = []
    regimes = list(res["regimes"].keys())
    lines.append("## B1. Success rates vs refinement steps (single checkpoint, fixed test set)\n")
    lines.append("| Refinement steps | lr | Any-candidate SR, % | Per-candidate SR, % | "
                 "Median per-pose success fraction, % | Diversity (successful), rad |")
    lines.append("|---|---|---|---|---|---|")
    pre = res["pre_refinement"]
    lines.append(f"| 0 | — | {pre['sr_any_pct']:.2f} | {pre['sr_per_candidate_pct']:.2f} | "
                 f"{pre['per_pose_success_frac']['median_pct']:.2f} | "
                 f"{res['diversity']['0']['diversity_successful_rad']:.2f} |")
    for r in regimes:
        m = res["regimes"][r]
        d = res["diversity"][r]
        lines.append(f"| {r} | {m['refine_lr']:g} | {m['sr_any_pct']:.2f} | "
                     f"{m['sr_per_candidate_pct']:.2f} | "
                     f"{m['per_pose_success_frac']['median_pct']:.2f} | "
                     f"{d['diversity_successful_rad']:.2f} |")
    lines.append("")
    lines.append("## B2. Distinct solution modes per pose (successful candidates)\n")
    lines.append("| Refinement steps | τ, rad | Mean modes | Median modes | Min–max | Poses evaluated |")
    lines.append("|---|---|---|---|---|---|")
    for r in regimes:
        for tau, cl in res["clustering"][r].items():
            lines.append(f"| {r} | {tau} | {cl['mean_modes']:.2f} | {cl['median_modes']:.0f} | "
                         f"{cl['min_modes']}–{cl['max_modes']} | {cl['poses_evaluated']} |")
    lines.append("")
    lines.append("## Residual statistics at regime endpoints\n")
    lines.append("| Refinement steps | Pos mean, mm | Pos median, mm | Pos P95, mm | "
                 "Ori mean, ° | Ori median, ° | Ori P95, ° | Out-of-limits, % | Time, s |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    def _row(tag, m, t):
        return (f"| {tag} | {m['pos_mm']['mean']:.3f} | {m['pos_mm']['median']:.3f} | "
                f"{m['pos_mm']['p95']:.3f} | {m['ori_deg']['mean']:.3f} | "
                f"{m['ori_deg']['median']:.3f} | {m['ori_deg']['p95']:.3f} | "
                f"{m.get('out_of_limits_pct', 0.0):.2f} | {t} |")
    lines.append(_row("0", pre, f"{res['timings']['sampling_s']:.0f} (sampling)"))
    for r in regimes:
        m = res["regimes"][r]
        lines.append(_row(r, m, f"{m['refine_time_s']:.0f}"))
    with open(os.path.join(outdir, "tables.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, default=None, help="path to model state_dict (.pt)")
    ap.add_argument("--random-init", action="store_true",
                    help="use random weights (pipeline smoke test only)")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--targets", type=int, default=500)
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=None,
                    help="targets per sampling chunk (default: 500 on GPU, 25 on CPU)")
    ap.add_argument("--regimes", type=str, default="50:0.01,200:0.005,500:0.003",
                    help="comma-separated steps:lr refinement regimes")
    ap.add_argument("--taus", type=str, default="0.25,0.5,1.0",
                    help="joint-space clustering thresholds, rad")
    ap.add_argument("--pos-thresh-mm", type=float, default=1.0)
    ap.add_argument("--ori-thresh-deg", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="results_block_b")
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke run: 8 targets, 10 samples, 30:0.01 regime")
    args = ap.parse_args()

    if args.quick:
        args.targets, args.samples = 8, 10
        args.regimes = "10:0.01,30:0.01"

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.chunk is None:
        args.chunk = 500 if device.type == "cuda" else 25
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    log(f"Device: {device}; targets={args.targets}, samples={args.samples}, "
        f"seed={args.seed}, chunk={args.chunk}")

    # model (paper's final config: ResMLP 1024x12, eps, cosine T=200, CFG 1.5)
    model = ResMLPDenoiser(jdim=7, cdim=12, hid=1024, layers=12).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ckpt_info = {"n_params": n_params}
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k: v.float() for k, v in sd.items()}  # accept fp16-saved weights
        model.load_state_dict(sd)
        with open(args.ckpt, "rb") as f:
            ckpt_info.update({
                "path": args.ckpt,
                "sha256": hashlib.sha256(f.read()).hexdigest(),
            })
        log(f"Loaded checkpoint {args.ckpt} ({n_params:,} params)")
    elif args.random_init:
        log("WARNING: --random-init — results are meaningless (pipeline test only)")
        ckpt_info["path"] = "RANDOM_INIT"
    else:
        ap.error("provide --ckpt PATH or --random-init")
    model.eval()

    ns = NoiseScheduler(T=200, schedule="cosine", device=device)
    norm = JointNormalizer(device)
    fk = FK(device)
    cfg = {"pred_type": "eps", "guidance_scale": 1.5, "sample_steps": 50}
    pos_thr = args.pos_thresh_mm / M2MM
    ori_thr = args.ori_thresh_deg / RAD2DEG
    lo = torch.tensor(JOINTS_LOWER_LIMIT, device=device)
    hi = torch.tensor(JOINTS_UPPER_LIMIT, device=device)
    taus = [float(t) for t in args.taus.split(",")]
    regimes = [(int(s.split(":")[0]), float(s.split(":")[1]))
               for s in args.regimes.split(",")]

    # 1) fixed test set
    q_true, R_t, t_t = make_test_set(args.targets, args.seed, device)
    log(f"Test set: {args.targets} reachable poses (seed {args.seed})")

    # 2) diffusion candidates
    log("Sampling candidates (DDIM 50 steps, CFG w=1.5)...")
    joints0, R_exp, t_exp, t_sample = sample_candidates(
        model, ns, norm, cfg, R_t, t_t, args.samples, args.chunk, device)

    with torch.no_grad():
        pos0, ori0 = task_errors(fk, joints0, R_exp, t_exp)
    succ0 = (pos0 < pos_thr) & (ori0 < ori_thr)
    pre = summarize_errors(pos0, ori0, pos_thr, ori_thr, args.targets, args.samples,
                           joints0, lo, hi)
    diversity = {"0": diversity_metrics(joints0, succ0, args.targets, args.samples)}
    log(f"Pre-refinement: any-SR={pre['sr_any_pct']:.2f}%, "
        f"per-candidate SR={pre['sr_per_candidate_pct']:.2f}%, "
        f"pos_mean={pre['pos_mm']['mean']:.1f}mm")

    # 3) refinement regimes (each from the same initial candidates)
    results_regimes, convergence, clustering = {}, {}, {}
    arrays = {
        "q_true": q_true.cpu().numpy(), "R_target": R_t.cpu().numpy(),
        "t_target": t_t.cpu().numpy(), "joints_initial": joints0.cpu().numpy(),
    }
    for steps, lr in regimes:
        key = str(steps)
        log(f"Refinement regime: {steps} steps @ lr={lr:g}")
        snaps, curve, t_ref = refine_with_curves(
            joints0, R_exp, t_exp, fk, steps, lr, snap_steps={steps}, device=device)
        jr = snaps[steps]
        with torch.no_grad():
            pos, ori = task_errors(fk, jr, R_exp, t_exp)
        succ = (pos < pos_thr) & (ori < ori_thr)
        m = summarize_errors(pos, ori, pos_thr, ori_thr, args.targets, args.samples,
                             jr, lo, hi)
        m.update({"refine_lr": lr, "refine_time_s": t_ref})
        results_regimes[key] = m
        convergence[key] = curve
        diversity[key] = diversity_metrics(jr, succ, args.targets, args.samples)
        clustering[key] = clustering_analysis(jr, succ, args.targets, args.samples, taus)
        arrays[f"joints_refined_{key}"] = jr.cpu().numpy()
        log(f"  -> any-SR={m['sr_any_pct']:.2f}%, per-candidate SR="
            f"{m['sr_per_candidate_pct']:.2f}%, "
            f"div={diversity[key]['diversity_successful_rad']:.2f} rad, "
            f"out-of-limits={m['out_of_limits_pct']:.2f}%")

    # 4) save everything
    res = {
        "config": vars(args) | {"device": str(device)},
        "checkpoint": ckpt_info,
        "pre_refinement": pre,
        "regimes": results_regimes,
        "diversity": diversity,
        "clustering": clustering,
        "convergence": convergence,
        "timings": {"sampling_s": t_sample},
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    import numpy as np
    np.savez_compressed(os.path.join(args.out, "arrays.npz"), **arrays)
    make_tables(res, args.out)
    make_figures(res, args.out)
    log(f"Saved results to {args.out}/ (results.json, arrays.npz, tables.md, figures)")


if __name__ == "__main__":
    main()
