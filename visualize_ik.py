#!/usr/bin/env python3
"""Generate IK solutions for fixed end-effector poses and show statistics."""

import torch
import torch.nn.functional as F
from math import pi
from diffusion_ik import (
    ResMLPDenoiser, NoiseScheduler, JointNormalizer,
    sample_loop, _refine, base, build_model
)
from fk import FK
from common import JOINTS_LOWER_LIMIT, JOINTS_UPPER_LIMIT


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    fk = FK(device)
    norm = JointNormalizer(device)
    ns = NoiseScheduler(200, "cosine", device)

    cfg = base(
        tag="viz", arch="resmlp", hidden=1024, layers=12,
        pred_type="eps", schedule="cosine", T=200,
        cfg_dropout=0.1, guidance_scale=1.5,
        sample_steps=50,
    )

    model, n_params = build_model(cfg, device)
    ckpt = "best_v14_cfg_baseline.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"Model: {n_params:,} params (loaded {ckpt})")

    # Generate a few fixed targets from known joint configs
    test_joints = torch.tensor([
        [0.0, 0.262, -3.142, 2.618, 0.0, 1.309, 0.0],         # "home" pose
        [-3.142, 0.262, -3.046, 2.316, -0.192, 2.499, 0.710],  # "forward" pose
        [1.0, -0.5, 0.8, 1.5, -1.0, 0.7, 0.3],                 # arbitrary pose 1
        [-0.5, 1.2, -1.0, 2.0, 0.5, 1.8, -0.5],                # arbitrary pose 2
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],                   # zero pose
    ], device=device)

    pose_names = ["Home", "Forward", "Arbitrary1", "Arbitrary2", "Zero"]

    with torch.no_grad():
        R_targets, t_targets = fk(test_joints)

    N_SAMPLES = 100  # samples per target
    REFINE_STEPS = [0, 50, 200]

    for i, name in enumerate(pose_names):
        R_tgt = R_targets[i:i+1]
        t_tgt = t_targets[i:i+1]
        gt_joints = test_joints[i]

        print(f"\n{'='*70}")
        print(f"  Target: {name}")
        print(f"  GT joints (rad): [{', '.join(f'{v:.3f}' for v in gt_joints.tolist())}]")
        print(f"  EE position (mm): [{', '.join(f'{v*1000:.1f}' for v in t_tgt.squeeze().tolist())}]")
        print(f"{'='*70}")

        # Generate samples from diffusion
        cond = torch.cat([R_tgt.reshape(1, 9), t_tgt.reshape(1, 3)], 1).expand(N_SAMPLES, -1)
        x_raw = sample_loop(model, ns, cond, device, cfg)
        joints_raw = norm.denormalize(x_raw.clamp(-1, 1))

        for rs in REFINE_STEPS:
            if rs == 0:
                joints = joints_raw
                label = "Raw diffusion"
            else:
                joints = _refine(
                    joints_raw.clone(),
                    R_tgt.expand(N_SAMPLES, -1, -1),
                    t_tgt.expand(N_SAMPLES, -1, -1),
                    fk, rs, 0.005
                )
                label = f"+ {rs} refine steps"

            # Compute FK for predicted joints
            with torch.no_grad():
                R_pred, t_pred = fk(joints)

            # Position error
            pos_err = (t_pred.squeeze(-1) - t_tgt.squeeze(-1)).norm(dim=-1)
            pos_mm = pos_err * 1000

            # Orientation error
            R_diff = torch.matmul(R_pred.transpose(-1, -2), R_tgt.expand_as(R_pred))
            trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
            ori_err = torch.acos(((trace - 1) / 2).clamp(-1, 1))
            ori_deg = ori_err * 180 / pi

            # Success (pos < 1mm AND ori < 1 deg)
            success = (pos_mm < 1.0) & (ori_deg < 1.0)
            sr = success.float().mean().item()

            # Joint diversity (mean pairwise distance)
            if joints.shape[0] >= 2:
                diff = joints.unsqueeze(0) - joints.unsqueeze(1)
                pw = diff.norm(dim=-1)
                mask = torch.triu(torch.ones_like(pw, dtype=torch.bool), diagonal=1)
                diversity = pw[mask].mean().item()
            else:
                diversity = 0.0

            print(f"\n  {label} ({N_SAMPLES} samples):")
            print(f"    Position error:    mean={pos_mm.mean():.2f}mm  "
                  f"median={pos_mm.median():.2f}mm  "
                  f"min={pos_mm.min():.3f}mm  max={pos_mm.max():.2f}mm  "
                  f"p95={pos_mm.quantile(0.95):.2f}mm")
            print(f"    Orientation error: mean={ori_deg.mean():.2f}deg  "
                  f"median={ori_deg.median():.2f}deg  "
                  f"min={ori_deg.min():.3f}deg  max={ori_deg.max():.2f}deg  "
                  f"p95={ori_deg.quantile(0.95):.2f}deg")
            print(f"    Success rate:      {sr:.0%} ({success.sum().item()}/{N_SAMPLES})")
            print(f"    Joint diversity:   {diversity:.4f} rad")

            # Show a few sample joint configs
            if rs == max(REFINE_STEPS):
                print(f"\n    Sample solutions (first 5):")
                for j in range(min(5, joints.shape[0])):
                    jv = joints[j].tolist()
                    pe = pos_mm[j].item()
                    oe = ori_deg[j].item()
                    print(f"      [{', '.join(f'{v:7.3f}' for v in jv)}]  "
                          f"pos={pe:.3f}mm ori={oe:.3f}deg")


if __name__ == "__main__":
    main()
