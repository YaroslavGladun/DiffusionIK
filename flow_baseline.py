#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flow_baseline.py — conditional normalizing-flow IK baseline (IKFlow-style)
for the unified xArm 7 benchmark (Block A of the paper revision).

A faithful, self-contained reimplementation of the IKFlow idea
(Ames et al., 2022): a conditional normalizing flow p(q | pose) built from
affine coupling layers, trained by exact maximum likelihood on the same
procedurally generated FK data used by DiffusionIK, evaluated on the same
500 seeded test poses with the same 1 mm / 1 deg thresholds and the same
diversity metric. No invertibility-free components are used — this is the
architectural regime the paper contrasts against.

Usage:
    uv run python flow_baseline.py train  [--steps 20000]
    uv run python flow_baseline.py eval   [--ckpt flow_xarm7.pt]
    uv run python flow_baseline.py all
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn

from fk import FK
from common import JOINTS_LOWER_LIMIT, JOINTS_UPPER_LIMIT
from eval_ik import geodesic_distance
from benchmark_block_a import (make_test_set, make_near_singular_set,
                               wilson_ci)

RAD2DEG = 180.0 / math.pi
M2MM = 1000.0
DIM = 7
CDIM = 13  # 12-D pose + 1-D SoftFlow noise magnitude (as in IKFlow)


class CouplingMLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d_out))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class AffineCoupling(nn.Module):
    """y_a = x_a; y_b = x_b * exp(s) + t, with (s, t) = f(x_a, cond)."""

    def __init__(self, mask, hidden=256):
        super().__init__()
        self.register_buffer('mask', mask)          # 1 = pass-through part
        n_a = int(mask.sum().item())
        n_b = DIM - n_a
        self.net = CouplingMLP(n_a + CDIM, 2 * n_b, hidden)
        self.n_b = n_b

    def forward(self, x, c, inverse=False):
        a = x[:, self.mask.bool()]
        st = self.net(torch.cat([a, c], dim=-1))
        s, t = st[:, :self.n_b], st[:, self.n_b:]
        s = torch.tanh(s) * 2.0                     # stabilise scaling
        y = x.clone()
        if not inverse:
            y[:, ~self.mask.bool()] = x[:, ~self.mask.bool()] * torch.exp(s) + t
            logdet = s.sum(dim=-1)
        else:
            y[:, ~self.mask.bool()] = (x[:, ~self.mask.bool()] - t) * torch.exp(-s)
            logdet = -s.sum(dim=-1)
        return y, logdet


class ConditionalRealNVP(nn.Module):
    def __init__(self, n_layers=12, hidden=256, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        layers = []
        for i in range(n_layers):
            perm = torch.randperm(DIM, generator=g)
            mask = torch.zeros(DIM)
            mask[perm[:DIM // 2 + (i % 2)]] = 1.0   # alternate 3/4 split
            layers.append(AffineCoupling(mask, hidden))
        self.layers = nn.ModuleList(layers)

    def forward(self, q, c):
        """q -> z with total log|det J|; used for NLL training."""
        z, total = q, torch.zeros(q.shape[0], device=q.device)
        for lay in self.layers:
            z, ld = lay(z, c, inverse=False)
            total = total + ld
        return z, total

    def inverse(self, z, c):
        x = z
        for lay in reversed(self.layers):
            x, _ = lay(x, c, inverse=True)
        return x

    def nll(self, q, c):
        z, logdet = self.forward(q, c)
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * DIM * math.log(2 * math.pi)
        return -(log_pz + logdet).mean()

    @torch.no_grad()
    def sample(self, c, temperature=1.0):
        z = torch.randn(c.shape[0], DIM, device=c.device) * temperature
        return self.inverse(z, c)


def normalizers(device):
    lo = torch.tensor(JOINTS_LOWER_LIMIT, device=device)
    hi = torch.tensor(JOINTS_UPPER_LIMIT, device=device)
    to_norm = lambda q: 2 * (q - lo) / (hi - lo) - 1
    from_norm = lambda x: ((x + 1) / 2 * (hi - lo) + lo).clamp(lo, hi)
    return lo, hi, to_norm, from_norm


def cond_vec(R, t, sigma=None):
    """Pose conditioning + SoftFlow noise magnitude. The solution set of a
    pose is a 1-D manifold in the 7-D joint space (measure zero), so plain
    MLE training leaves the base-Gaussian mass spread far from it. SoftFlow
    (used by IKFlow for exactly this reason) perturbs the data with noise of
    a random magnitude and conditions the flow on that magnitude; sampling
    with the smallest magnitude then concentrates mass near the manifold."""
    B = R.shape[0]
    if sigma is None:
        sigma = torch.zeros(B, 1, device=R.device)
    return torch.cat([R.reshape(B, 9), t.reshape(B, 3), sigma * 10.0], dim=1)


def train(args, device):
    fk = FK(device)
    lo, hi, to_norm, _ = normalizers(device)
    model = ConditionalRealNVP(args.layers, args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'flow params: {n_params:,}')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05,
        anneal_strategy='cos', div_factor=10, final_div_factor=100)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        with torch.no_grad():
            q = torch.rand(args.batch, DIM, device=device) * (hi - lo) + lo
            R, t = fk(q)
            x = to_norm(q)
            # SoftFlow: log-uniform noise magnitude, injected into x and c
            logs = (torch.rand(args.batch, 1, device=device) *
                    (math.log10(args.soft_max) - math.log10(args.soft_min)) +
                    math.log10(args.soft_min))
            sigma = 10.0 ** logs
            x = x + sigma * torch.randn_like(x)
            c = cond_vec(R, t, sigma)
        loss = model.nll(x, c)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % max(1, args.steps // 40) == 0 or step == 1:
            print(f'  [{step:6d}/{args.steps}] nll={loss.item():.4f} '
                  f'lr={opt.param_groups[0]["lr"]:.2e} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    torch.save({'state_dict': model.state_dict(),
                'config': {'layers': args.layers, 'hidden': args.hidden,
                           'steps': args.steps, 'batch': args.batch,
                           'lr': args.lr, 'n_params': n_params}},
               args.ckpt)
    print(f'saved {args.ckpt} ({time.time() - t0:.0f}s total)')


@torch.no_grad()
def evaluate(args, device):
    fk = FK(device)
    _, _, _, from_norm = normalizers(device)
    ck = torch.load(args.ckpt, map_location=device)
    model = ConditionalRealNVP(ck['config']['layers'], ck['config']['hidden']).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()

    results = {}
    for set_name in (('main',) if not args.singular_too else ('main', 'singular')):
        if set_name == 'main':
            _, R_t, t_t = make_test_set(args.targets, 0, device)
        else:
            _, R_t, t_t, _ = make_near_singular_set(args.targets, 0, device)
        n, m_samp = R_t.shape[0], args.samples
        Rx = R_t.repeat_interleave(m_samp, 0)
        tx = t_t.repeat_interleave(m_samp, 0)
        sig = torch.full((Rx.shape[0], 1), args.soft_min, device=device)
        t0 = time.time()
        q = from_norm(model.sample(cond_vec(Rx, tx, sig)).clamp(-1, 1))
        wall = time.time() - t0
        R_p, t_p = fk(q)
        pos = (t_p - tx).squeeze(-1).norm(dim=-1)
        ori = geodesic_distance(R_p, Rx)
        succ = (pos < 1e-3) & (ori < math.pi / 180)
        succ_pose = succ.view(n, m_samp)
        sr_any = succ_pose.any(1).float().mean().item() * 100
        div = []
        qv = q.view(n, m_samp, DIM)
        for i in range(n):
            qs = qv[i][succ_pose[i]]
            if qs.shape[0] >= 2:
                div.append(torch.pdist(qs).mean().item())
        div_all = [torch.pdist(qv[i]).mean().item() for i in range(n)]
        results[set_name] = {
            'SR_any_pct': sr_any,
            'SR_per_candidate_pct': succ.float().mean().item() * 100,
            'SR_wilson95': wilson_ci(int(succ_pose.any(1).sum()), n),
            'pos_mm': {'mean': pos.mean().item() * M2MM,
                       'median': pos.median().item() * M2MM,
                       'p95': pos.quantile(0.95).item() * M2MM},
            'ori_deg': {'mean': ori.mean().item() * RAD2DEG,
                        'median': ori.median().item() * RAD2DEG,
                        'p95': ori.quantile(0.95).item() * RAD2DEG},
            'time_ms_per_query_batch50': wall / n * 1000,
            'solutions_per_target': m_samp,
            'diversity_successful_rad': sum(div) / len(div) if div else None,
            'diversity_all_rad': sum(div_all) / len(div_all),
            'n_params': ck['config']['n_params'],
            'train_config': ck['config'],
        }
        print(set_name, json.dumps(results[set_name])[:400])
    out = 'results_block_a/flow_results.json'
    os.makedirs('results_block_a', exist_ok=True)
    json.dump(results, open(out, 'w'), indent=2)
    print('saved', out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['train', 'eval', 'all'])
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--layers', type=int, default=12)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--soft-min', type=float, default=1e-3)
    ap.add_argument('--soft-max', type=float, default=0.2)
    ap.add_argument('--ckpt', type=str, default='flow_xarm7.pt')
    ap.add_argument('--targets', type=int, default=500)
    ap.add_argument('--samples', type=int, default=50)
    ap.add_argument('--singular-too', action='store_true')
    ap.add_argument('--device', type=str, default=None)
    args = ap.parse_args()
    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(0)
    print('device:', device)
    if args.mode in ('train', 'all'):
        train(args, device)
    if args.mode in ('eval', 'all'):
        evaluate(args, device)


if __name__ == '__main__':
    main()
