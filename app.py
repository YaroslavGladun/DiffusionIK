#!/usr/bin/env python3
"""
app.py -- Interactive 3D visualization of diffusion-based IK for xArm7.

Run:  uv run python app.py
Open: http://localhost:7860
"""

import numpy as np
import torch
import trimesh
import plotly.graph_objects as go
import gradio as gr
from math import pi
from pathlib import Path

from diffusion_ik import (
    base, build_model, JointNormalizer, NoiseScheduler,
    sample_loop, _refine,
)
from fk import FK
from common import TransformationUtility, JOINTS_LOWER_LIMIT, JOINTS_UPPER_LIMIT

# =====================================================================
# Global Setup
# =====================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = base(
    tag="app", arch="resmlp", hidden=1024, layers=12,
    pred_type="eps", schedule="cosine", T=200,
    cfg_dropout=0.1, guidance_scale=1.5, sample_steps=50,
)
model, n_params = build_model(cfg, DEVICE)
model.load_state_dict(
    torch.load("best_v14_cfg_baseline.pt", map_location=DEVICE, weights_only=True)
)
model.eval()
print(f"Model: {n_params:,} params on {DEVICE}")

ns = NoiseScheduler(200, "cosine", DEVICE)
norm = JointNormalizer(DEVICE)
fk = FK(DEVICE)

# Pre-compute fixed joint transforms as numpy (mirrors fk.py:12-20)
DH_PARAMS = torch.tensor([
    [0, 0, 0.267, 0, 0, 0],
    [0, 0, 0, -1.5708, 0, 0],
    [0, -0.293, 0, 1.5708, 0, 0],
    [0.0525, 0, 0, 1.5708, 0, 0],
    [0.0775, -0.3425, 0, 1.5708, 0, 0],
    [0, 0, 0, 1.5708, 0, 0],
    [0.076, 0.097, 0, -1.5708, 0, 0],
])
_R_fixed, _t_fixed = TransformationUtility.xyz_rpy_to_torch_affine(DH_PARAMS)
R_FIXED_NP = _R_fixed.numpy()        # (7, 3, 3)
T_FIXED_NP = _t_fixed.squeeze(-1).numpy()  # (7, 3)

# Load & optionally decimate STL meshes
STL_DIR = Path(
    "/home/yaroslav/development/bcr_azure_ml_py/submodules/"
    "bcr_description/xarm_description/meshes/xarm7/visual"
)
STL_NAMES = ["link_base.STL"] + [f"link{i}.STL" for i in range(1, 8)]
MAX_FACES = 2000


def _load_mesh(path):
    m = trimesh.load(path, force="mesh")
    if len(m.faces) > MAX_FACES:
        try:
            m = m.simplify_quadric_decimation(MAX_FACES)
        except Exception:
            pass
    return m


MESHES = [_load_mesh(STL_DIR / n) for n in STL_NAMES]
print(f"Loaded {len(MESHES)} meshes "
      f"({sum(len(m.faces) for m in MESHES):,} total faces)")


# =====================================================================
# Per-link FK (numpy, mirrors fk.py forward() but captures every frame)
# =====================================================================

def fk_all_links(joints_7):
    """Return 8 frames [(R,t), ...] for link_base + link1..link7."""
    frames = [(np.eye(3), np.zeros(3))]
    R = np.eye(3)
    t = np.zeros(3)
    for i in range(7):
        t = R @ T_FIXED_NP[i] + t
        R = R @ R_FIXED_NP[i]
        c, s = np.cos(joints_7[i]), np.sin(joints_7[i])
        Rj = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        R = R @ Rj
        frames.append((R.copy(), t.copy()))
    return frames


# =====================================================================
# Plotly helpers
# =====================================================================

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
]


def build_robot_traces(joints, idx, color):
    """Mesh3d traces for one arm configuration."""
    frames = fk_all_links(joints)
    traces = []
    for li, (mesh, (R, t)) in enumerate(zip(MESHES, frames)):
        v = (R @ np.asarray(mesh.vertices).T).T + t
        f = np.asarray(mesh.faces)
        traces.append(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=color,
            opacity=0.7 if idx == 0 else 0.4,
            name=f"Sol {idx + 1}" if li == 0 else None,
            showlegend=(li == 0),
            legendgroup=f"sol_{idx}",
            hoverinfo="skip",
        ))
    return traces


# =====================================================================
# Callbacks
# =====================================================================

def generate_and_visualize(x, y, z, roll, pitch, yaw, n_solutions, refine_steps):
    N = int(n_solutions)
    rs = int(refine_steps)

    # Target rotation (Rz @ Ry @ Rx, same as TransformationUtility)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R_tgt_np = Rz @ Ry @ Rx
    t_tgt_np = np.array([x, y, z])

    R_tgt = torch.tensor(R_tgt_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    t_tgt = torch.tensor(t_tgt_np, dtype=torch.float32, device=DEVICE).reshape(1, 3, 1)

    cond = torch.cat([R_tgt.reshape(1, 9), t_tgt.reshape(1, 3)], 1).expand(N, -1)

    with torch.no_grad():
        xn = sample_loop(model, ns, cond, DEVICE, cfg)
        joints = norm.denormalize(xn.clamp(-1, 1))

    if rs > 0:
        joints = _refine(
            joints,
            R_tgt.expand(N, -1, -1),
            t_tgt.expand(N, -1, -1),
            fk, rs, 0.005,
        )

    # Errors
    with torch.no_grad():
        Rp, tp = fk(joints)
        pos_err = (tp.squeeze(-1) - t_tgt.squeeze(-1)).norm(dim=-1)
        Rd = torch.matmul(Rp.transpose(-1, -2), R_tgt.expand_as(Rp))
        tr = Rd.diagonal(dim1=-2, dim2=-1).sum(-1)
        ori_err = torch.acos(((tr - 1) / 2).clamp(-1, 1))

    pos_mm = pos_err.cpu().numpy() * 1000
    ori_deg = ori_err.cpu().numpy() * 180 / pi
    joints_np = joints.detach().cpu().numpy()
    sort_idx = np.argsort(pos_mm + ori_deg * 0.1)

    # Build figure
    fig = go.Figure()
    for rank, si in enumerate(sort_idx):
        fig.add_traces(build_robot_traces(
            joints_np[si], rank, COLORS[rank % len(COLORS)],
        ))

    # Target marker
    fig.add_trace(go.Scatter3d(
        x=[t_tgt_np[0]], y=[t_tgt_np[1]], z=[t_tgt_np[2]],
        mode="markers",
        marker=dict(size=8, color="red", symbol="diamond"),
        name="Target",
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=700,
        legend=dict(x=0.02, y=0.98),
    )

    # Info text
    lines = [f"{'#':>3}  {'pos(mm)':>8}  {'ori(deg)':>8}"]
    lines.append("-" * 28)
    for rank, si in enumerate(sort_idx):
        lines.append(f"{rank + 1:3d}  {pos_mm[si]:8.3f}  {ori_deg[si]:8.3f}")
    return fig, "\n".join(lines)


def random_reachable_target():
    lo = torch.tensor(JOINTS_LOWER_LIMIT, device=DEVICE)
    hi = torch.tensor(JOINTS_UPPER_LIMIT, device=DEVICE)
    j = torch.rand(1, 7, device=DEVICE) * (hi - lo) + lo
    with torch.no_grad():
        R, t = fk(j)
    rpy = TransformationUtility.rotation_matrix_to_rpy(R)
    tv = t.squeeze().cpu().tolist()
    rv = rpy.squeeze().cpu().tolist()
    return tv[0], tv[1], tv[2], rv[0], rv[1], rv[2]


# =====================================================================
# Gradio UI
# =====================================================================

with gr.Blocks(title="xArm7 Diffusion IK") as demo:
    gr.Markdown("# xArm7 Diffusion IK Visualizer")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Target Pose")
            x_sl = gr.Slider(-0.7, 0.7, value=0.206, step=0.001, label="X (m)")
            y_sl = gr.Slider(-0.7, 0.7, value=0.0, step=0.001, label="Y (m)")
            z_sl = gr.Slider(-0.2, 0.8, value=0.121, step=0.001, label="Z (m)")
            roll_sl = gr.Slider(-pi, pi, value=0.0, step=0.01, label="Roll (rad)")
            pitch_sl = gr.Slider(-pi, pi, value=0.0, step=0.01, label="Pitch (rad)")
            yaw_sl = gr.Slider(-pi, pi, value=0.0, step=0.01, label="Yaw (rad)")

            gr.Markdown("### Settings")
            n_sl = gr.Slider(1, 16, value=8, step=1, label="Solutions")
            refine_sl = gr.Slider(0, 300, value=100, step=10, label="Refine Steps")

            gen_btn = gr.Button("Generate", variant="primary")
            rand_btn = gr.Button("Random Reachable Target")
            info_box = gr.Textbox(label="Results", lines=20, interactive=False)

        with gr.Column(scale=2):
            plot = gr.Plot(label="3D View")

    gen_btn.click(
        fn=generate_and_visualize,
        inputs=[x_sl, y_sl, z_sl, roll_sl, pitch_sl, yaw_sl, n_sl, refine_sl],
        outputs=[plot, info_box],
    )
    rand_btn.click(
        fn=random_reachable_target,
        outputs=[x_sl, y_sl, z_sl, roll_sl, pitch_sl, yaw_sl],
    )

if __name__ == "__main__":
    demo.launch()
