#!/usr/bin/env python3
import torch
from math import pi
from diffusion_ik import (
    base, build_model, JointNormalizer, NoiseScheduler,
    sample_loop, _refine,
)
from fk import FK

device = torch.device("cuda")
fk = FK(device)
norm = JointNormalizer(device)
ns = NoiseScheduler(200, "cosine", device)

cfg = base(tag="viz", arch="resmlp", hidden=1024, layers=12,
           pred_type="eps", schedule="cosine", T=200,
           cfg_dropout=0.1, guidance_scale=1.5, sample_steps=50)

model, _ = build_model(cfg, device)
model.load_state_dict(torch.load("best_v14_cfg_baseline.pt", map_location=device, weights_only=True))
model.eval()

# Fixed target: "Forward" pose
gt = torch.tensor([[-3.142, 0.262, -3.046, 2.316, -0.192, 2.499, 0.710]], device=device)
with torch.no_grad():
    R_tgt, t_tgt = fk(gt)

print(f"Target EE pos (mm): [{t_tgt[0,0,0]*1000:.1f}, {t_tgt[0,1,0]*1000:.1f}, {t_tgt[0,2,0]*1000:.1f}]")
print()

N = 20
cond = torch.cat([R_tgt.reshape(1, 9), t_tgt.reshape(1, 3)], 1).expand(N, -1)
x = sample_loop(model, ns, cond, device, cfg)
joints = norm.denormalize(x.clamp(-1, 1))

# Refine 200 steps
joints = _refine(joints, R_tgt.expand(N, -1, -1), t_tgt.expand(N, -1, -1), fk, 200, 0.005)

with torch.no_grad():
    Rp, tp = fk(joints)
    pos_err = (tp.squeeze(-1) - t_tgt.squeeze(-1)).norm(dim=-1) * 1000
    Rd = torch.matmul(Rp.transpose(-1, -2), R_tgt.expand_as(Rp))
    tr = Rd.diagonal(dim1=-2, dim2=-1).sum(-1)
    ori_err = torch.acos(((tr - 1) / 2).clamp(-1, 1)) * 180 / pi

header = f"{'#':>3}  {'j1':>7} {'j2':>7} {'j3':>7} {'j4':>7} {'j5':>7} {'j6':>7} {'j7':>7}   {'pos(mm)':>8}  {'ori(deg)':>8}"
print(header)
print("-" * len(header))
for i in range(N):
    j = joints[i].tolist()
    print(f"{i+1:3d}  {j[0]:7.3f} {j[1]:7.3f} {j[2]:7.3f} {j[3]:7.3f} {j[4]:7.3f} {j[5]:7.3f} {j[6]:7.3f}   {pos_err[i]:8.3f}  {ori_err[i]:8.3f}")
