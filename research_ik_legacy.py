#!/usr/bin/env python3
"""
research_ik.py — Iterative IK model research with wandb logging.

Usage:
    uv run python research_ik.py --tag v5 --model cond_j0 --hidden-dim 256 --num-layers 12 --epochs 600
"""

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import argparse
from math import pi

from fk import FK
from common import JointValuesScalerInverse
from eval_ik import evaluate_ik


# --------------- models ---------------

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        return x + F.silu(self.bn(self.fc(x)))


class IKModelFull7(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=6, latent_dim=0):
        super().__init__()
        self.latent_dim = latent_dim
        self.proj = nn.Linear(12 + latent_dim, hidden_dim)
        self.proj_bn = nn.BatchNorm1d(hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_layers)])
        self.out_norm = nn.BatchNorm1d(hidden_dim)
        self.head_sin = nn.Linear(hidden_dim, 7)
        self.head_cos = nn.Linear(hidden_dim, 7)

    def forward(self, x):
        x = F.silu(self.proj_bn(self.proj(x)))
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return torch.atan2(self.head_sin(x), self.head_cos(x))


class IKModelCondJ0(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=8):
        super().__init__()
        d_half = hidden_dim // 2
        self.fc1 = nn.Linear(14, d_half)
        self.bn1 = nn.BatchNorm1d(d_half)
        self.fc2 = nn.Linear(d_half, d_half)
        self.bn2 = nn.BatchNorm1d(d_half)
        self.skip_fcs = nn.ModuleList()
        self.skip_bns = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.skip_fcs.append(nn.Linear(hidden_dim, d_half))
            self.skip_bns.append(nn.BatchNorm1d(d_half))
        self.head_sin = nn.Linear(d_half, 6)
        self.head_cos = nn.Linear(d_half, 6)
        self.act = nn.SiLU()

    def forward(self, x):
        x1 = self.act(self.bn1(self.fc1(x)))
        x2 = self.act(self.bn2(self.fc2(x1)))
        prev2, prev1 = x1, x2
        for fc, bn in zip(self.skip_fcs, self.skip_bns):
            out = self.act(bn(fc(torch.cat([prev2, prev1], dim=1))))
            prev2, prev1 = prev1, out
        return torch.atan2(self.head_sin(prev1), self.head_cos(prev1))


# --------------- helpers ---------------

def generate_epoch_data(fk, scaler_inv, total_samples, device):
    """One big FK call for the whole epoch."""
    with torch.no_grad():
        joints = scaler_inv(torch.rand(total_samples, 7, device=device))
        joints = torch.atan2(torch.sin(joints), torch.cos(joints))
        R, t = fk(joints)
    return joints, R, t


def geodesic_error(R_pred, R_target):
    R_diff = torch.matmul(R_pred.transpose(-1, -2), R_target)
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))


def make_input_full7(R, t, latent_dim, device):
    B = R.shape[0]
    parts = [R.reshape(B, 9), t.reshape(B, 3)]
    if latent_dim > 0:
        parts.append(torch.randn(B, latent_dim, device=device))
    return torch.cat(parts, dim=1)


def make_input_cond_j0(j0, R, t):
    B = R.shape[0]
    return torch.cat([torch.sin(j0).reshape(B, 1),
                      torch.cos(j0).reshape(B, 1),
                      R.reshape(B, 9),
                      t.reshape(B, 3)], dim=1)


# --------------- training ---------------

def train_one_epoch(model, fk, scaler_inv, optimizer, scheduler, cfg, device):
    model.train()
    bs = cfg["batch_size"]
    n_batches = cfg["batches_per_epoch"]
    model_type = cfg["model"]
    pos_sum = rot_sum = 0.0

    # generate all data in one FK call
    joints_all, R_all, t_all = generate_epoch_data(
        fk, scaler_inv, bs * n_batches, device)

    for i in range(n_batches):
        s, e = i * bs, (i + 1) * bs
        R_tgt, t_tgt = R_all[s:e], t_all[s:e]
        batch_joints = joints_all[s:e]

        if model_type == "cond_j0":
            x = make_input_cond_j0(batch_joints[:, 0], R_tgt, t_tgt)
            pred_6 = model(x)
            pred_joints = torch.cat([batch_joints[:, :1], pred_6], dim=1)
        else:
            x = make_input_full7(R_tgt, t_tgt, cfg["latent_dim"], device)
            pred_joints = model(x)

        R_pred, t_pred = fk(pred_joints)

        pos_loss = torch.mean(torch.norm(
            t_pred.squeeze(-1) - t_tgt.squeeze(-1), dim=-1))
        rot_loss = torch.mean(torch.norm(
            (R_pred - R_tgt).reshape(-1, 9), dim=-1))
        loss = cfg["alpha"] * rot_loss + cfg["beta"] * pos_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()  # OneCycleLR: step per batch

        pos_sum += pos_loss.item()
        rot_sum += rot_loss.item()

    return pos_sum / n_batches, rot_sum / n_batches


# --------------- evaluation ---------------

@torch.no_grad()
def quick_eval(model, fk, scaler_inv, device, cfg, n=10000):
    model.eval()
    joints, R_tgt, t_tgt = generate_epoch_data(fk, scaler_inv, n, device)

    if cfg["model"] == "cond_j0":
        x = make_input_cond_j0(joints[:, 0], R_tgt, t_tgt)
        pred_6 = model(x)
        pred_joints = torch.cat([joints[:, :1], pred_6], dim=1)
    else:
        x = make_input_full7(R_tgt, t_tgt, 0, device)
        pred_joints = model(x)

    R_pred, t_pred = fk(pred_joints)
    pos_err = torch.norm(t_pred.squeeze(-1) - t_tgt.squeeze(-1), dim=-1)
    ori_err = geodesic_error(R_pred, R_tgt)

    return {
        "quick/pos_mean_mm": pos_err.mean().item() * 1000,
        "quick/ori_mean_deg": ori_err.mean().item() * 180 / pi,
        "quick/pos_p95_mm": pos_err.quantile(0.95).item() * 1000,
        "quick/ori_p95_deg": ori_err.quantile(0.95).item() * 180 / pi,
        "quick/pos_p99_mm": pos_err.quantile(0.99).item() * 1000,
        "quick/ori_p99_deg": ori_err.quantile(0.99).item() * 180 / pi,
    }


def refine_joints(joints, R_tgt, t_tgt, fk, steps, lr):
    """Gradient-based refinement through differentiable FK."""
    joints = joints.detach().clone().requires_grad_(True)
    optim = torch.optim.Adam([joints], lr=lr)
    n = joints.shape[0]
    with torch.enable_grad():
        for _ in range(steps):
            R_pred, t_pred = fk(joints)
            pos_loss = torch.mean(torch.norm(
                t_pred.squeeze(-1) - t_tgt.squeeze(-1), dim=-1))
            rot_loss = torch.mean(torch.norm(
                (R_pred - R_tgt).reshape(n, 9), dim=-1))
            loss = pos_loss + rot_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
    return joints.detach()


def make_predict_fn(model, device, cfg):
    fk_ref = FK(device)
    r_steps = cfg.get("refine_steps", 0)
    r_lr = cfg.get("refine_lr", 0.01)

    def predict_full7(R_tgt, t_tgt, n):
        model.eval()
        x = make_input_full7(
            R_tgt.expand(n, -1, -1), t_tgt.expand(n, -1, -1), 0, device)
        joints = model(x)
        if r_steps > 0:
            joints = refine_joints(
                joints, R_tgt.expand(n, -1, -1), t_tgt.expand(n, -1, -1),
                fk_ref, r_steps, r_lr)
        return joints

    def predict_cond_j0(R_tgt, t_tgt, n):
        model.eval()
        j0 = torch.linspace(-pi, pi, n, device=device)
        R_exp = R_tgt.expand(n, -1, -1)
        t_exp = t_tgt.expand(n, -1, -1)
        x = make_input_cond_j0(j0, R_exp, t_exp)
        with torch.no_grad():
            pred_6 = model(x)
        joints = torch.cat([j0.unsqueeze(1), pred_6], dim=1)
        if r_steps > 0:
            joints = refine_joints(joints, R_exp, t_exp, fk_ref, r_steps, r_lr)
        return joints

    return predict_cond_j0 if cfg["model"] == "cond_j0" else predict_full7


def save_iteration(tag, cfg, results, path="iterations.json"):
    log = []
    if os.path.exists(path):
        with open(path) as f:
            log = json.load(f)
    log.append({"tag": tag, "config": cfg, "results": results})
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


# --------------- main ---------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["full7", "cond_j0"], default="cond_j0")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--latent-dim", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=10000)
    p.add_argument("--batches-per-epoch", type=int, default=500)
    p.add_argument("--alpha", type=float, default=1.0, help="rotation loss weight")
    p.add_argument("--beta", type=float, default=10.0, help="position loss weight")
    p.add_argument("--eval-targets", type=int, default=500)
    p.add_argument("--eval-samples", type=int, default=50)
    p.add_argument("--pos-thresh", type=float, default=0.001, help="meters")
    p.add_argument("--ori-thresh", type=float, default=1.0, help="degrees")
    p.add_argument("--refine-steps", type=int, default=0, help="gradient refinement steps at eval")
    p.add_argument("--refine-lr", type=float, default=0.01, help="refinement step size")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--resume", type=str, default="", help="path to .pt checkpoint")
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--notes", type=str, default="")
    args = p.parse_args()

    cfg = vars(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wandb.init(project="DiffusionIK", config=cfg, name=args.tag, notes=args.notes)

    if args.model == "cond_j0":
        model = IKModelCondJ0(args.hidden_dim, args.num_layers).to(device)
    else:
        model = IKModelFull7(args.hidden_dim, args.num_layers, args.latent_dim).to(device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        print(f"Resumed from {args.resume}")

    num_params = sum(p.numel() for p in model.parameters())
    wandb.config.update({"num_params": num_params}, allow_val_change=True)
    print(f"Model: {args.model} | Parameters: {num_params:,}")

    fk = FK(device)
    scaler_inv = JointValuesScalerInverse(device)

    total_steps = args.epochs * args.batches_per_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.05,        # 5% warmup
        anneal_strategy='cos',
        div_factor=10,         # start_lr = max_lr / 10
        final_div_factor=100,  # end_lr = start_lr / 100
    )

    predict_fn = make_predict_fn(model, device, cfg)
    ori_thresh_rad = args.ori_thresh * pi / 180
    best_sr = 0.0

    for epoch in range(1, args.epochs + 1):
        pos_loss, rot_loss = train_one_epoch(
            model, fk, scaler_inv, optimizer, scheduler, cfg, device
        )

        lr_now = optimizer.param_groups[0]['lr']
        log = {
            "epoch": epoch,
            "train/pos_loss": pos_loss,
            "train/rot_loss": rot_loss,
            "train/total_loss": cfg["alpha"] * rot_loss + cfg["beta"] * pos_loss,
            "lr": lr_now,
        }
        log.update(quick_eval(model, fk, scaler_inv, device, cfg))

        do_full_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if do_full_eval:
            results = evaluate_ik(
                predict_fn, device,
                num_targets=args.eval_targets,
                samples_per_target=args.eval_samples,
                position_threshold_m=args.pos_thresh,
                orientation_threshold_rad=ori_thresh_rad,
            )
            log["eval/success_rate"] = results["success_rate"]
            log["eval/diversity"] = results["diversity"]
            sr = results["success_rate"]
            if sr > best_sr:
                best_sr = sr
                torch.save(model.state_dict(), f"best_{args.tag}.pt")
            print(
                f"[Epoch {epoch}] SR={sr:.2%} div={results['diversity']:.4f} "
                f"pos_p99={log['quick/pos_p99_mm']:.2f}mm "
                f"ori_p99={log['quick/ori_p99_deg']:.2f}deg "
                f"lr={lr_now:.2e}"
            )
        else:
            print(
                f"[Epoch {epoch}] pos={pos_loss:.6f} rot={rot_loss:.6f} "
                f"pos_p99={log['quick/pos_p99_mm']:.2f}mm "
                f"ori_p99={log['quick/ori_p99_deg']:.2f}deg "
                f"lr={lr_now:.2e}"
            )

        wandb.log(log)

    torch.save(model.state_dict(), f"final_{args.tag}.pt")
    save_iteration(args.tag, cfg, {
        "best_success_rate": best_sr,
        "num_params": num_params,
    })
    wandb.config.update({"best_success_rate": best_sr}, allow_val_change=True)
    print(f"\nBest success rate: {best_sr:.2%}")
    wandb.finish()


if __name__ == "__main__":
    main()
