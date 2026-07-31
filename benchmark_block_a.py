#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_block_a.py — Unified IK benchmark on the xArm 7 (paper revision, Block A).

Every method is evaluated on the SAME seeded set of reachable target poses
(identical to analysis_block_b.py: seed=0, poses sampled in joint space and
mapped through the differentiable FK), with the SAME success thresholds
(1 mm position, 1 deg orientation), and reports the SAME metric set:

    SR | mean / median / P95 position error | mean orientation error |
    per-query latency | solutions per target | diversity

Methods implemented here:
    dls          — damped least squares (Levenberg–Marquardt on the 6-D task
                   residual, numerical Jacobian, random restarts)
    dls-singular — the same solver on a NEAR-SINGULAR pose set (low
                   manipulability index), for the robustness analysis

Rows for the learning-based methods (regression baseline, IKFlow, DiffusionIK)
are produced by their own scripts / checkpoints and merged into the same
results file; this harness defines the protocol and the numerical baseline.

Usage:
    uv run python benchmark_block_a.py --methods dls
    uv run python benchmark_block_a.py --methods dls,dls-singular --out results_block_a
"""

import argparse
import json
import os
import time
from math import pi

import torch

from fk import FK
from common import JOINTS_LOWER_LIMIT, JOINTS_UPPER_LIMIT
from eval_ik import geodesic_distance

RAD2DEG = 180.0 / pi
M2MM = 1000.0


# ─────────────────────────────────────────────────────────────
# Test sets (identical protocol to analysis_block_b.make_test_set)
# ─────────────────────────────────────────────────────────────

def make_test_set(n_targets, seed, device):
    g = torch.Generator(device='cpu').manual_seed(seed)
    lo = torch.tensor(JOINTS_LOWER_LIMIT)
    hi = torch.tensor(JOINTS_UPPER_LIMIT)
    q = torch.rand(n_targets, 7, generator=g) * (hi - lo) + lo
    q = q.to(device)
    fk = FK(device)
    with torch.no_grad():
        R, t = fk(q)
    return q, R, t


def manipulability(fk, q):
    """Yoshikawa index w = sqrt(det(J J^T)) from the numerical 6xn Jacobian."""
    J = numerical_jacobian(fk, q)                      # (B, 6, 7)
    JJt = J @ J.transpose(1, 2)
    return torch.linalg.det(JJt).clamp(min=0).sqrt()


def make_near_singular_set(n_targets, seed, device, pool_factor=40):
    """Sample a large pool of reachable poses, keep the lowest-manipulability
    fraction. These are the 'challenging scenarios' requested by Reviewer 3."""
    g = torch.Generator(device='cpu').manual_seed(seed + 777)
    lo = torch.tensor(JOINTS_LOWER_LIMIT)
    hi = torch.tensor(JOINTS_UPPER_LIMIT)
    pool = torch.rand(n_targets * pool_factor, 7, generator=g) * (hi - lo) + lo
    pool = pool.to(device)
    fk = FK(device)
    w = manipulability(fk, pool)
    idx = torch.argsort(w)[:n_targets]
    q = pool[idx]
    with torch.no_grad():
        R, t = fk(q)
    return q, R, t, w[idx]


# ─────────────────────────────────────────────────────────────
# Task-space residual and Jacobian
# ─────────────────────────────────────────────────────────────

def task_residual(fk, q, R_t, t_t):
    """6-D residual [e_pos; e_rot] with e_rot = log-map of R_t R(q)^T."""
    R_c, t_c = fk(q)
    e_pos = (t_t - t_c).squeeze(-1)                          # (B, 3)
    R_err = R_t @ R_c.transpose(1, 2)                        # (B, 3, 3)
    tr = R_err.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_th = ((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    th = torch.acos(cos_th)                                  # (B,)
    skew = torch.stack([R_err[:, 2, 1] - R_err[:, 1, 2],
                        R_err[:, 0, 2] - R_err[:, 2, 0],
                        R_err[:, 1, 0] - R_err[:, 0, 1]], dim=-1)  # (B, 3)
    # e_rot = th * axis; axis = skew / (2 sin th); safe near th -> 0
    scale = torch.where(th > 1e-6, th / (2 * torch.sin(th)),
                        torch.full_like(th, 0.5))
    e_rot = scale.unsqueeze(-1) * skew
    return torch.cat([e_pos, e_rot], dim=-1)                 # (B, 6)


def numerical_jacobian(fk, q, h=1e-6):
    """d(residual)/dq by forward differences, batched over poses.
    Residual here is the pose itself (position + orientation increment),
    so J is the standard geometric Jacobian up to sign."""
    B = q.shape[0]
    R0, t0 = fk(q)
    cols = []
    for j in range(7):
        qj = q.clone()
        qj[:, j] += h
        Rj, tj = fk(qj)
        dpos = (tj - t0).squeeze(-1) / h                     # (B, 3)
        dR = Rj @ R0.transpose(1, 2)
        drot = torch.stack([dR[:, 2, 1] - dR[:, 1, 2],
                            dR[:, 0, 2] - dR[:, 2, 0],
                            dR[:, 1, 0] - dR[:, 0, 1]], dim=-1) / (2 * h)
        cols.append(torch.cat([dpos, drot], dim=-1))         # (B, 6)
    return torch.stack(cols, dim=-1)                         # (B, 6, 7)


# ─────────────────────────────────────────────────────────────
# Damped least squares with random restarts
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def dls_solve_batch(fk, R_t, t_t, q0, lo, hi, iters=150, lam=0.05):
    """One DLS run from q0 for a batch of poses. Returns final q."""
    q = q0.clone()
    I6 = torch.eye(6, device=q.device).unsqueeze(0)
    active = torch.ones(q.shape[0], dtype=torch.bool, device=q.device)
    for _ in range(iters):
        if not active.any():
            break
        e = task_residual(fk, q[active], R_t[active], t_t[active])   # (A, 6)
        J = numerical_jacobian(fk, q[active])                        # (A, 6, 7)
        A = J @ J.transpose(1, 2) + (lam ** 2) * I6
        y = torch.linalg.solve(A, e.unsqueeze(-1))                   # (A, 6, 1)
        dq = (J.transpose(1, 2) @ y).squeeze(-1)                     # (A, 7)
        qa = (q[active] + dq).clamp(lo, hi)
        q[active] = qa
        pos = e[:, :3].norm(dim=-1)
        rot = e[:, 3:].norm(dim=-1)
        done = (pos < 0.5e-3) & (rot < 0.5 * pi / 180)   # tighten to be safe
        idx = active.nonzero(as_tuple=True)[0]
        active[idx[done]] = False
    return q


def run_dls(fk, R_t, t_t, lo, hi, pos_thr, ori_thr, seed,
            max_restarts=10, iters=150, lam=0.05):
    """DLS with random restarts until success or budget exhaustion."""
    n = R_t.shape[0]
    device = R_t.device
    g = torch.Generator(device='cpu').manual_seed(seed + 12345)
    best_q = torch.zeros(n, 7, device=device)
    best_err = torch.full((n,), float('inf'), device=device)
    solved = torch.zeros(n, dtype=torch.bool, device=device)
    restarts_used = torch.zeros(n, dtype=torch.long, device=device)

    t0 = time.time()
    for r in range(max_restarts):
        todo = ~solved
        if not todo.any():
            break
        q0 = (torch.rand(int(todo.sum()), 7, generator=g) *
              (hi - lo).cpu() + lo.cpu()).to(device)
        q_fin = dls_solve_batch(fk, R_t[todo], t_t[todo], q0, lo, hi,
                                iters=iters, lam=lam)
        pos, ori = final_errors(fk, q_fin, R_t[todo], t_t[todo])
        ok = (pos < pos_thr) & (ori < ori_thr)
        score = pos + ori  # combined, only used to keep the best attempt
        idx = todo.nonzero(as_tuple=True)[0]
        better = score < best_err[idx]
        best_err[idx[better]] = score[better]
        best_q[idx[better]] = q_fin[better]
        newly = idx[ok]
        solved[newly] = True
        restarts_used[idx[~ok]] += 1
    wall = time.time() - t0
    return best_q, solved, restarts_used, wall


def final_errors(fk, q, R_t, t_t):
    R_c, t_c = fk(q)
    pos = (t_c - t_t).squeeze(-1).norm(dim=-1)
    ori = geodesic_distance(R_c, R_t)
    return pos, ori


# ─────────────────────────────────────────────────────────────
# Metrics in the unified Table format
# ─────────────────────────────────────────────────────────────

def table_metrics(pos, ori, solved, wall_per_query_ms, solutions_per_target,
                  diversity, extra=None):
    m = {
        'SR_pct': solved.float().mean().item() * 100,
        'pos_mm': {'mean': pos.mean().item() * M2MM,
                   'median': pos.median().item() * M2MM,
                   'p95': pos.quantile(0.95).item() * M2MM},
        'ori_deg': {'mean': ori.mean().item() * RAD2DEG,
                    'median': ori.median().item() * RAD2DEG,
                    'p95': ori.quantile(0.95).item() * RAD2DEG},
        'time_ms_per_query': wall_per_query_ms,
        'solutions_per_target': solutions_per_target,
        'diversity': diversity,
    }
    if extra:
        m.update(extra)
    return m


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (max(0.0, center - half) * 100, min(1.0, center + half) * 100)


# ─────────────────────────────────────────────────────────────
# Single-query latency (realistic per-call timing, batch of 1)
# ─────────────────────────────────────────────────────────────

def measure_single_query_latency(fk, R_t, t_t, lo, hi, pos_thr, ori_thr,
                                 seed, n_queries=50, **dls_kw):
    times = []
    for i in range(min(n_queries, R_t.shape[0])):
        t0 = time.perf_counter()
        run_dls(fk, R_t[i:i + 1], t_t[i:i + 1], lo, hi, pos_thr, ori_thr,
                seed + i, **dls_kw)
        times.append((time.perf_counter() - t0) * 1000)
    ts = torch.tensor(times)
    return {'median_ms': ts.median().item(), 'mean_ms': ts.mean().item(),
            'p95_ms': ts.quantile(0.95).item()}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--methods', type=str, default='dls')
    ap.add_argument('--targets', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--pos-thresh-mm', type=float, default=1.0)
    ap.add_argument('--ori-thresh-deg', type=float, default=1.0)
    ap.add_argument('--max-restarts', type=int, default=10)
    ap.add_argument('--iters', type=int, default=150)
    ap.add_argument('--lam', type=float, default=0.05)
    ap.add_argument('--latency-queries', type=int, default=50)
    ap.add_argument('--out', type=str, default='results_block_a')
    args = ap.parse_args()

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    fk = FK(device)
    lo = torch.tensor(JOINTS_LOWER_LIMIT, device=device)
    hi = torch.tensor(JOINTS_UPPER_LIMIT, device=device)
    pos_thr = args.pos_thresh_mm / M2MM
    ori_thr = args.ori_thresh_deg / RAD2DEG

    results_path = os.path.join(args.out, 'results.json')
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    results.setdefault('protocol', {
        'targets': args.targets, 'seed': args.seed,
        'pos_thresh_mm': args.pos_thresh_mm,
        'ori_thresh_deg': args.ori_thresh_deg,
        'platform_note': 'DLS timed on CPU (single process); '
                         'learning-based methods timed on GPU as noted per row.',
    })

    for method in args.methods.split(','):
        method = method.strip()
        print(f'=== {method} ===', flush=True)

        if method in ('dls', 'dls-singular'):
            if method == 'dls':
                q_true, R_t, t_t = make_test_set(args.targets, args.seed, device)
                extra_set = {}
            else:
                q_true, R_t, t_t, w = make_near_singular_set(
                    args.targets, args.seed, device)
                extra_set = {'manipulability': {
                    'mean': w.mean().item(), 'median': w.median().item()}}
                print(f'  near-singular set: manipulability median '
                      f'{w.median().item():.2e}')

            best_q, solved, restarts, wall = run_dls(
                fk, R_t, t_t, lo, hi, pos_thr, ori_thr, args.seed,
                max_restarts=args.max_restarts, iters=args.iters, lam=args.lam)
            pos, ori = final_errors(fk, best_q, R_t, t_t)
            lat = measure_single_query_latency(
                fk, R_t, t_t, lo, hi, pos_thr, ori_thr, args.seed,
                n_queries=args.latency_queries,
                max_restarts=args.max_restarts, iters=args.iters, lam=args.lam)
            n = R_t.shape[0]
            m = table_metrics(
                pos, ori, solved, lat['median_ms'], 1, 'No',
                extra={'batch_wall_s': wall,
                       'latency': lat,
                       'restarts': {'mean': restarts.float().mean().item(),
                                    'max': int(restarts.max().item())},
                       'SR_wilson95': wilson_ci(int(solved.sum()), n),
                       'solved_only': {
                           'pos_mm': {
                               'mean': pos[solved].mean().item() * M2MM,
                               'median': pos[solved].median().item() * M2MM,
                               'p95': pos[solved].quantile(0.95).item() * M2MM},
                           'ori_deg': {
                               'mean': ori[solved].mean().item() * RAD2DEG,
                               'median': ori[solved].median().item() * RAD2DEG,
                               'p95': ori[solved].quantile(0.95).item() * RAD2DEG}},
                       'solver': {'iters': args.iters, 'lam': args.lam,
                                  'max_restarts': args.max_restarts},
                       **extra_set})
            results[method] = m
            import numpy as np
            np.savez_compressed(
                os.path.join(args.out, f'perpose_{method}.npz'),
                pos_m=pos.cpu().numpy(), ori_rad=ori.cpu().numpy(),
                solved=solved.cpu().numpy(),
                restarts=restarts.cpu().numpy(), q_best=best_q.cpu().numpy())
            print(json.dumps(m, indent=2)[:600])
        else:
            print(f'  method {method!r} is provided by its own script; skipping')

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # markdown rows for the paper table
    md = ['| Method | SR, % | Mean pos, mm | Median pos, mm | P95 pos, mm | '
          'Mean ori, ° | Time/query, ms | Solutions | Diversity |',
          '|---|---|---|---|---|---|---|---|---|']
    for name, m in results.items():
        if name == 'protocol':
            continue
        md.append(f"| {name} | {m['SR_pct']:.2f} | {m['pos_mm']['mean']:.3f} | "
                  f"{m['pos_mm']['median']:.3f} | {m['pos_mm']['p95']:.3f} | "
                  f"{m['ori_deg']['mean']:.3f} | {m['time_ms_per_query']:.1f} | "
                  f"{m['solutions_per_target']} | {m['diversity']} |")
    with open(os.path.join(args.out, 'table.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('saved:', results_path)


if __name__ == '__main__':
    main()
