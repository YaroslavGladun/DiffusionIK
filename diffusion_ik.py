#!/usr/bin/env python3
"""
diffusion_ik.py — Diffusion-based Inverse Kinematics for 7-DOF xArm.

Iterative research: 15 adaptive experiments with wandb logging.
Architectures: MLP, ResMLP, Transformer, DiT.
Techniques: eps/x0/v prediction, DDIM sampling, CFG, FK auxiliary loss.

Usage:
    uv run python diffusion_ik.py
"""

import math
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from math import pi

from fk import FK
from common import JOINTS_LOWER_LIMIT, JOINTS_UPPER_LIMIT
from eval_ik import evaluate_ik


def log(msg):
    print(msg, flush=True)


# ═══════════════════════════════════════════════════════════
# Joint Normalization (angles -> [-1, 1] for diffusion)
# ═══════════════════════════════════════════════════════════

class JointNormalizer:
    def __init__(self, device):
        self.lo = torch.tensor(JOINTS_LOWER_LIMIT, device=device)
        self.hi = torch.tensor(JOINTS_UPPER_LIMIT, device=device)
        self.rng = self.hi - self.lo

    def normalize(self, joints):
        return 2.0 * (joints - self.lo) / self.rng - 1.0

    def denormalize(self, x):
        return ((x + 1.0) / 2.0 * self.rng + self.lo).clamp(self.lo, self.hi)


# ═══════════════════════════════════════════════════════════
# Noise Scheduler (DDPM forward + DDIM reverse)
# ═══════════════════════════════════════════════════════════

def cosine_beta_schedule(T, s=0.008):
    t = torch.linspace(0, T, T + 1, dtype=torch.float64)
    ab = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    ab = ab / ab[0]
    betas = 1 - ab[1:] / ab[:-1]
    return betas.clamp(0, 0.999).float()


def linear_beta_schedule(T, lo=1e-4, hi=0.02):
    return torch.linspace(lo, hi, T)


class NoiseScheduler:
    def __init__(self, T=200, schedule="cosine", device="cpu"):
        self.T = T
        self.device = device
        betas = cosine_beta_schedule(T) if schedule == "cosine" else linear_beta_schedule(T)
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.ab = torch.cumprod(self.alphas, 0)
        self.sqrt_ab = self.ab.sqrt()
        self.sqrt_1m = (1 - self.ab).sqrt()

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return self.sqrt_ab[t, None] * x0 + self.sqrt_1m[t, None] * noise, noise

    def pred_x0(self, xt, t, out, pred_type):
        a, b = self.sqrt_ab[t, None], self.sqrt_1m[t, None]
        if pred_type == "eps":
            return (xt - b * out) / a.clamp(min=1e-6)
        if pred_type == "x0":
            return out
        return a * xt - b * out  # v-prediction

    def get_target(self, x0, noise, t, pred_type):
        if pred_type == "eps":
            return noise
        if pred_type == "x0":
            return x0
        a, b = self.sqrt_ab[t, None], self.sqrt_1m[t, None]
        return a * noise - b * x0


# ═══════════════════════════════════════════════════════════
# Sinusoidal Time Embedding
# ═══════════════════════════════════════════════════════════

class SinTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = math.log(10000) / (half - 1)
        f = torch.exp(torch.arange(half, device=t.device) * -f)
        e = t.float()[:, None] * f[None, :]
        return torch.cat([e.sin(), e.cos()], -1)


# ═══════════════════════════════════════════════════════════
# Denoiser Architectures
# ═══════════════════════════════════════════════════════════

# ---- MLP Denoiser (time injected at every layer) ----

class _MLPBlock(nn.Module):
    def __init__(self, dim, tdim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.tp = nn.Linear(tdim, dim)

    def forward(self, x, te):
        return F.silu(self.ln(self.fc(x) + self.tp(te)))


class MLPDenoiser(nn.Module):
    def __init__(self, jdim=7, cdim=12, hid=512, layers=8, tdim=128):
        super().__init__()
        self.temb = nn.Sequential(
            SinTimeEmb(tdim), nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim),
        )
        self.inp = nn.Linear(jdim + cdim, hid)
        self.blks = nn.ModuleList([_MLPBlock(hid, tdim) for _ in range(layers)])
        self.out = nn.Linear(hid, jdim)

    def forward(self, x, t, c):
        te = self.temb(t)
        h = self.inp(torch.cat([x, c], -1))
        for b in self.blks:
            h = b(h, te)
        return self.out(h)


# ---- ResMLP Denoiser ----

class _ResBlock(nn.Module):
    def __init__(self, dim, tdim):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.tp = nn.Linear(tdim, dim)

    def forward(self, x, te):
        h = F.silu(self.fc1(self.n1(x))) + self.tp(te)
        return x + self.fc2(self.n2(h))


class ResMLPDenoiser(nn.Module):
    def __init__(self, jdim=7, cdim=12, hid=512, layers=8, tdim=128):
        super().__init__()
        self.temb = nn.Sequential(
            SinTimeEmb(tdim), nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim),
        )
        self.inp = nn.Linear(jdim + cdim, hid)
        self.blks = nn.ModuleList([_ResBlock(hid, tdim) for _ in range(layers)])
        self.ln = nn.LayerNorm(hid)
        self.out = nn.Linear(hid, jdim)

    def forward(self, x, t, c):
        te = self.temb(t)
        h = self.inp(torch.cat([x, c], -1))
        for b in self.blks:
            h = b(h, te)
        return self.out(self.ln(h))


# ---- Transformer / DiT Denoiser ----

class _AdaLN(nn.Module):
    def __init__(self, dim, cdim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cdim, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, c):
        s, b = self.proj(c).chunk(2, -1)
        if x.dim() == 3:
            s, b = s.unsqueeze(1), b.unsqueeze(1)
        return self.norm(x) * (1 + s) + b


class _DiTBlock(nn.Module):
    def __init__(self, dim, heads, ffm=4, cdim=None):
        super().__init__()
        self.aln1 = _AdaLN(dim, cdim or dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.aln2 = _AdaLN(dim, cdim or dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * ffm), nn.GELU(), nn.Linear(dim * ffm, dim))

    def forward(self, x, c):
        h = self.aln1(x, c)
        h, _ = self.attn(h, h, h)
        x = x + h
        return x + self.ff(self.aln2(x, c))


class TransformerDenoiser(nn.Module):
    def __init__(self, jdim=7, cdim=12, dm=256, heads=4, layers=6,
                 tdim=128, adaln=False, ffm=4):
        super().__init__()
        self.adaln = adaln
        self.temb = nn.Sequential(
            SinTimeEmb(tdim), nn.Linear(tdim, dm), nn.SiLU(), nn.Linear(dm, dm),
        )
        self.je = nn.Linear(1, dm)
        self.jp = nn.Parameter(torch.randn(1, jdim, dm) * 0.02)
        self.cp = nn.Linear(cdim, dm)
        if adaln:
            self.merge = nn.Sequential(nn.Linear(dm * 2, dm), nn.SiLU(), nn.Linear(dm, dm))
            self.blks = nn.ModuleList([_DiTBlock(dm, heads, ffm, dm) for _ in range(layers)])
        else:
            enc = nn.TransformerEncoderLayer(
                dm, heads, dm * ffm, batch_first=True, activation="gelu", norm_first=True,
            )
            self.tf = nn.TransformerEncoder(enc, layers)
        self.ln = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, 1)

    def forward(self, x, t, c):
        te = self.temb(t)
        jtok = self.je(x.unsqueeze(-1)) + self.jp
        ce = self.cp(c)
        if self.adaln:
            ac = self.merge(torch.cat([te, ce], -1))
            h = jtok
            for b in self.blks:
                h = b(h, ac)
        else:
            seq = torch.cat([te.unsqueeze(1), ce.unsqueeze(1), jtok], 1)
            h = self.tf(seq)[:, 2:]
        return self.head(self.ln(h)).squeeze(-1)


# ═══════════════════════════════════════════════════════════
# Model Factory
# ═══════════════════════════════════════════════════════════

def build_model(cfg, device):
    j, cd = 7, cfg.get("cond_dim", 12)
    a = cfg["arch"]
    if a == "mlp":
        m = MLPDenoiser(j, cd, cfg.get("hidden", 512), cfg.get("layers", 8))
    elif a == "resmlp":
        m = ResMLPDenoiser(j, cd, cfg.get("hidden", 512), cfg.get("layers", 8))
    elif a == "transformer":
        m = TransformerDenoiser(j, cd, cfg.get("dm", 256), cfg.get("heads", 4),
                                cfg.get("layers", 6), adaln=False, ffm=cfg.get("ffm", 4))
    elif a == "dit":
        m = TransformerDenoiser(j, cd, cfg.get("dm", 256), cfg.get("heads", 4),
                                cfg.get("layers", 6), adaln=True, ffm=cfg.get("ffm", 4))
    else:
        raise ValueError(a)
    m = m.to(device)
    n = sum(p.numel() for p in m.parameters())
    assert n <= 400_000_000, f"{n:,} params > 400M limit"
    return m, n


# ═══════════════════════════════════════════════════════════
# Data Generation
# ═══════════════════════════════════════════════════════════

def gen_data(fk, norm, N, device):
    with torch.no_grad():
        lo = torch.tensor(JOINTS_LOWER_LIMIT, device=device)
        hi = torch.tensor(JOINTS_UPPER_LIMIT, device=device)
        j = torch.rand(N, 7, device=device) * (hi - lo) + lo
        R, t = fk(j)
        return norm.normalize(j), torch.cat([R.reshape(N, 9), t.reshape(N, 3)], 1)


# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════

def train_epoch(model, opt, sched_lr, ns, fk, norm, device, cfg):
    model.train()
    bs, nb = cfg["batch_size"], cfg["batches_per_epoch"]
    pt = cfg["pred_type"]
    x0_all, cond_all = gen_data(fk, norm, bs * nb, device)
    cd = cfg.get("cfg_dropout", 0.0)
    fw = cfg.get("fk_loss_weight", 0.0)
    loss_sum = 0.0

    for i in range(nb):
        s, e = i * bs, (i + 1) * bs
        x0, cond = x0_all[s:e], cond_all[s:e]
        t = torch.randint(0, ns.T, (bs,), device=device)
        noise = torch.randn_like(x0)
        xt, _ = ns.q_sample(x0, t, noise)
        target = ns.get_target(x0, noise, t, pt)

        if cd > 0:
            mask = (torch.rand(bs, 1, device=device) < cd).float()
            cond = cond * (1 - mask)

        pred = model(xt, t, cond)
        loss = F.mse_loss(pred, target)

        if fw > 0:
            x0p = ns.pred_x0(xt, t, pred, pt).clamp(-1, 1)
            jp = norm.denormalize(x0p)
            Rp, tp = fk(jp)
            Rt = cond_all[s:e, :9].reshape(bs, 3, 3)
            tt = cond_all[s:e, 9:].reshape(bs, 3, 1)
            loss = loss + fw * (F.mse_loss(tp, tt) + F.mse_loss(Rp, Rt))

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched_lr is not None:
            sched_lr.step()
        loss_sum += loss.item()

    return loss_sum / nb


# ═══════════════════════════════════════════════════════════
# DDIM Sampling
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def sample_loop(model, ns, cond, device, cfg):
    model.eval()
    B = cond.shape[0]
    x = torch.randn(B, 7, device=device)
    pt = cfg["pred_type"]
    gs = cfg.get("guidance_scale", 0.0)
    steps = cfg.get("sample_steps", 50)

    # DDIM timestep schedule (ascending, deduplicated)
    ts = torch.linspace(0, ns.T - 1, steps + 1).round().long()
    ts = sorted(set(ts.tolist()))

    for i in range(len(ts) - 1, 0, -1):
        tv, tp = ts[i], ts[i - 1]
        tb = torch.full((B,), tv, device=device, dtype=torch.long)

        if gs > 0:
            pc = model(x, tb, cond)
            pu = model(x, tb, torch.zeros_like(cond))
            pred = pu + (1 + gs) * (pc - pu)
        else:
            pred = model(x, tb, cond)

        x0p = ns.pred_x0(x, tb, pred, pt).clamp(-1, 1)
        ab_t = ns.ab[tv]
        ab_p = ns.ab[tp]
        eps = (x - ab_t.sqrt() * x0p) / (1 - ab_t).sqrt().clamp(min=1e-8)
        x = ab_p.sqrt() * x0p + (1 - ab_p).sqrt() * eps

    return x


# ═══════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def quick_eval(model, ns, norm, fk, device, cfg, n=5000):
    model.eval()
    lo = torch.tensor(JOINTS_LOWER_LIMIT, device=device)
    hi = torch.tensor(JOINTS_UPPER_LIMIT, device=device)
    jr = torch.rand(n, 7, device=device) * (hi - lo) + lo
    Rt, tt = fk(jr)
    cond = torch.cat([Rt.reshape(n, 9), tt.reshape(n, 3)], 1)

    xp = sample_loop(model, ns, cond, device, cfg)
    jp = norm.denormalize(xp.clamp(-1, 1))
    Rp, tp = fk(jp)

    pos_err = (tp.squeeze(-1) - tt.squeeze(-1)).norm(dim=-1)
    Rd = torch.matmul(Rp.transpose(-1, -2), Rt)
    tr = Rd.diagonal(dim1=-2, dim2=-1).sum(-1)
    ori_err = torch.acos(((tr - 1) / 2).clamp(-1, 1))

    return {
        "q/pos_mm": pos_err.mean().item() * 1000,
        "q/pos_p95": pos_err.quantile(0.95).item() * 1000,
        "q/ori_deg": ori_err.mean().item() * 180 / pi,
        "q/ori_p95": ori_err.quantile(0.95).item() * 180 / pi,
    }


def make_predict_fn(model, ns, norm, device, cfg):
    fk_ref = FK(device)
    rs = cfg.get("refine_steps", 0)
    rlr = cfg.get("refine_lr", 0.01)

    def predict(R_tgt, t_tgt, n):
        cond = torch.cat([R_tgt.reshape(1, 9), t_tgt.reshape(1, 3)], 1).expand(n, -1)
        xn = sample_loop(model, ns, cond, device, cfg)
        joints = norm.denormalize(xn.clamp(-1, 1))
        if rs > 0:
            joints = _refine(joints, R_tgt.expand(n, -1, -1),
                             t_tgt.expand(n, -1, -1), fk_ref, rs, rlr)
        return joints

    return predict


def _refine(joints, Rt, tt, fk, steps, lr):
    j = joints.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([j], lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            Rp, tp = fk(j)
            loss = F.mse_loss(tp, tt) + F.mse_loss(Rp, Rt)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return j.detach()


# ═══════════════════════════════════════════════════════════
# Single Experiment Runner
# ═══════════════════════════════════════════════════════════

def run_experiment(cfg, device):
    tag = cfg["tag"]
    log(f"\n{'=' * 60}")
    log(f"  {tag}: {cfg.get('notes', '')}")
    log(f"{'=' * 60}")

    wandb.init(project="DiffusionIK", name=tag, config=cfg,
               notes=cfg.get("notes", ""), reinit=True, group="diffusion_v3")

    model, n_params = build_model(cfg, device)
    log(f"  arch={cfg['arch']} params={n_params:,}")
    wandb.config.update({"n_params": n_params}, allow_val_change=True)

    ns = NoiseScheduler(cfg["T"], cfg.get("schedule", "cosine"), device)
    fk = FK(device)
    norm = JointNormalizer(device)

    total_steps = cfg["epochs"] * cfg["batches_per_epoch"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 1e-4))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=total_steps,
        pct_start=0.05, anneal_strategy="cos", div_factor=10, final_div_factor=100,
    )

    predict_fn = make_predict_fn(model, ns, norm, device, cfg)
    best_sr = 0.0
    best_pos = 1e9
    t0 = time.time()

    for ep in range(1, cfg["epochs"] + 1):
        loss = train_epoch(model, opt, sched, ns, fk, norm, device, cfg)
        lr_now = opt.param_groups[0]["lr"]
        wlog = {"epoch": ep, "train/loss": loss, "lr": lr_now}

        do_quick = ep % cfg.get("qeval_every", 30) == 0 or ep == cfg["epochs"]
        do_full = ep % cfg.get("feval_every", 75) == 0 or ep == cfg["epochs"]

        msg = f"  [{ep:3d}] loss={loss:.5f} lr={lr_now:.1e}"

        if do_quick:
            qe = quick_eval(model, ns, norm, fk, device, cfg)
            wlog.update(qe)
            best_pos = min(best_pos, qe["q/pos_mm"])
            msg = (f"  [{ep:3d}] loss={loss:.4f} pos={qe['q/pos_mm']:.1f}mm "
                   f"ori={qe['q/ori_deg']:.1f}deg lr={lr_now:.1e}")

        if do_full:
            res = evaluate_ik(predict_fn, device,
                              num_targets=cfg.get("eval_targets", 500),
                              samples_per_target=50,
                              position_threshold_m=0.001,
                              orientation_threshold_rad=1.0 * pi / 180)
            wlog["eval/sr"] = res["success_rate"]
            wlog["eval/div"] = res["diversity"]
            if res["success_rate"] > best_sr:
                best_sr = res["success_rate"]
                torch.save(model.state_dict(), f"best_{tag}.pt")
            msg = (f"  [{ep:3d}] loss={loss:.4f} SR={res['success_rate']:.2%} "
                   f"div={res['diversity']:.4f}")

        log(msg)
        wandb.log(wlog)

    elapsed = time.time() - t0
    wandb.config.update({"best_sr": best_sr, "best_pos_mm": best_pos,
                          "time_s": elapsed}, allow_val_change=True)
    wandb.finish()
    log(f"  => SR={best_sr:.2%} pos={best_pos:.1f}mm | {elapsed:.0f}s | {n_params:,} params")
    return {"success_rate": best_sr, "pos_mm": best_pos, "tag": tag,
            "n_params": n_params, "time": elapsed, "loss": loss}


# ═══════════════════════════════════════════════════════════
# Config Helpers
# ═══════════════════════════════════════════════════════════

def base(**kw):
    c = dict(
        T=200, schedule="cosine", pred_type="eps",
        epochs=150, batch_size=4096, batches_per_epoch=300,
        lr=5e-4, wd=1e-4,
        sample_steps=50, guidance_scale=0.0,
        cfg_dropout=0.0, fk_loss_weight=0.0,
        refine_steps=0, refine_lr=0.01,
        qeval_every=30, feval_every=75,
        eval_targets=500, cond_dim=12,
    )
    c.update(kw)
    return c


def pick_best(R, keys):
    """Pick best by SR, break ties by pos_mm (lower is better)."""
    return min(keys, key=lambda k: (-R[k]["success_rate"], R[k].get("pos_mm", 1e9)))


# ═══════════════════════════════════════════════════════════
# Iterative Research (15 experiments)
# ═══════════════════════════════════════════════════════════

def run_all(device):
    R = {}

    # ── Phase 1: Architecture & Size ───────────────────────
    log("\n" + "#" * 60)
    log("  PHASE 1: Architecture & Model Size")
    log("#" * 60)

    R["v01"] = run_experiment(base(
        tag="v01_resmlp_s", arch="resmlp", hidden=256, layers=6,
        notes="ResMLP small (256, 6L)"), device)

    R["v02"] = run_experiment(base(
        tag="v02_resmlp_m", arch="resmlp", hidden=512, layers=8,
        notes="ResMLP medium (512, 8L)"), device)

    R["v03"] = run_experiment(base(
        tag="v03_resmlp_l", arch="resmlp", hidden=1024, layers=12,
        notes="ResMLP large (1024, 12L)"), device)

    R["v04"] = run_experiment(base(
        tag="v04_mlp_m", arch="mlp", hidden=512, layers=8,
        notes="MLP medium (512, 8L) for comparison"), device)

    bk = pick_best(R, ["v01", "v02", "v03", "v04"])
    acfg = {
        "v01": dict(arch="resmlp", hidden=256, layers=6),
        "v02": dict(arch="resmlp", hidden=512, layers=8),
        "v03": dict(arch="resmlp", hidden=1024, layers=12),
        "v04": dict(arch="mlp", hidden=512, layers=8),
    }[bk]
    log(f"\n>>> Best arch: {bk} SR={R[bk]['success_rate']:.2%} pos={R[bk]['pos_mm']:.1f}mm\n")

    # ── Phase 2: Prediction Target ─────────────────────────
    log("#" * 60)
    log("  PHASE 2: Prediction Target")
    log("#" * 60)

    R["v05"] = run_experiment(base(
        tag="v05_x0pred", pred_type="x0",
        notes="x0 prediction", **acfg), device)

    R["v06"] = run_experiment(base(
        tag="v06_vpred", pred_type="v",
        notes="v-prediction", **acfg), device)

    pmap = {bk: "eps", "v05": "x0", "v06": "v"}
    pk = pick_best(R, pmap)
    best_pred = pmap[pk]
    log(f"\n>>> Best pred: {best_pred} from {pk} SR={R[pk]['success_rate']:.2%} pos={R[pk]['pos_mm']:.1f}mm\n")

    # ── Phase 3: Schedule & Steps ──────────────────────────
    log("#" * 60)
    log("  PHASE 3: Schedule & Diffusion Steps")
    log("#" * 60)

    R["v07"] = run_experiment(base(
        tag="v07_linear", schedule="linear", pred_type=best_pred,
        notes="Linear beta schedule", **acfg), device)

    R["v08"] = run_experiment(base(
        tag="v08_T50", T=50, sample_steps=50, pred_type=best_pred,
        notes="T=50 (minimal diffusion steps)", **acfg), device)

    smap = {pk: ("cosine", 200), "v07": ("linear", 200), "v08": ("cosine", 50)}
    sk = pick_best(R, smap)
    best_sched, best_T = smap[sk]
    log(f"\n>>> Best schedule: {best_sched} T={best_T} from {sk} "
        f"SR={R[sk]['success_rate']:.2%} pos={R[sk]['pos_mm']:.1f}mm\n")

    # ── Phase 4: Advanced Techniques ───────────────────────
    log("#" * 60)
    log("  PHASE 4: Advanced Techniques")
    log("#" * 60)

    R["v09"] = run_experiment(base(
        tag="v09_cfg", pred_type=best_pred, schedule=best_sched, T=best_T,
        cfg_dropout=0.1, guidance_scale=1.5,
        notes="Classifier-free guidance", **acfg), device)

    R["v10"] = run_experiment(base(
        tag="v10_fkloss", pred_type=best_pred, schedule=best_sched, T=best_T,
        fk_loss_weight=0.1,
        notes="FK auxiliary loss", **acfg), device)

    R["v11"] = run_experiment(base(
        tag="v11_fkloss_strong", pred_type=best_pred, schedule=best_sched, T=best_T,
        fk_loss_weight=1.0,
        notes="FK auxiliary loss (strong)", **acfg), device)

    advM = {sk: {}, "v09": dict(cfg_dropout=0.1, guidance_scale=1.5),
            "v10": dict(fk_loss_weight=0.1), "v11": dict(fk_loss_weight=1.0)}
    ak = pick_best(R, advM)
    best_adv = advM[ak]
    log(f"\n>>> Best technique: {ak} SR={R[ak]['success_rate']:.2%} pos={R[ak]['pos_mm']:.1f}mm\n")

    # ── Phase 5: Final Optimization ────────────────────────
    log("#" * 60)
    log("  PHASE 5: Final Optimization")
    log("#" * 60)

    R["v12"] = run_experiment(base(
        tag="v12_long", pred_type=best_pred, schedule=best_sched, T=best_T,
        epochs=400, batches_per_epoch=500, lr=1e-3,
        qeval_every=50, feval_every=100,
        notes="Long training (200K steps)", **acfg, **best_adv), device)

    R["v13"] = run_experiment(base(
        tag="v13_long_hrlr", pred_type=best_pred, schedule=best_sched, T=best_T,
        epochs=400, batches_per_epoch=500, lr=3e-3, batch_size=8192,
        qeval_every=50, feval_every=100,
        notes="Long + higher LR + bigger batch", **acfg, **best_adv), device)

    fk2 = pick_best(R, ["v12", "v13", ak])
    fextra = {}
    if fk2 == "v12":
        fextra = dict(epochs=400, batches_per_epoch=500, lr=1e-3,
                       qeval_every=50, feval_every=100)
    elif fk2 == "v13":
        fextra = dict(epochs=400, batches_per_epoch=500, lr=3e-3, batch_size=8192,
                       qeval_every=50, feval_every=100)

    # ── Phase 6: Gradient Refinement ───────────────────────
    log("#" * 60)
    log("  PHASE 6: Gradient Refinement at Inference")
    log("#" * 60)

    R["v14"] = run_experiment(base(
        tag="v14_refine50", pred_type=best_pred, schedule=best_sched, T=best_T,
        refine_steps=50, refine_lr=0.01,
        notes="Refinement 50 steps", **acfg, **best_adv, **fextra), device)

    R["v15"] = run_experiment(base(
        tag="v15_refine200", pred_type=best_pred, schedule=best_sched, T=best_T,
        refine_steps=200, refine_lr=0.005,
        notes="Refinement 200 steps (more precise)", **acfg, **best_adv, **fextra), device)

    # ── Summary ────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("  RESEARCH SUMMARY")
    log("=" * 60)
    for k in sorted(R):
        r = R[k]
        log(f"  {r['tag']:25s} SR={r['success_rate']:6.2%}  pos={r['pos_mm']:7.1f}mm  "
            f"params={r['n_params']:>12,}  time={r['time']:5.0f}s")
    bst = pick_best(R, R)
    log(f"\n  BEST: {R[bst]['tag']} SR={R[bst]['success_rate']:.2%} pos={R[bst]['pos_mm']:.1f}mm")
    log("=" * 60)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    run_all(device)
