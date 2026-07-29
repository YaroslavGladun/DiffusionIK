# Diffusion-Based Inverse Kinematics: Experiment Log

## Problem Statement

Conditional diffusion model for 7-DOF xArm inverse kinematics: given target end-effector pose (rotation matrix R + translation t = 12D condition), generate joint angle configurations (7D) via iterative denoising.

## Setup

- **Robot**: xArm 7-DOF, joint limits from `common.py`
- **Forward Kinematics**: Differentiable FK in `fk.py` (DH parameters, batched)
- **Data**: Infinite on-the-fly generation — sample random joints, run FK to get (R, t) targets
- **Normalization**: Joint angles mapped to [-1, 1] for diffusion via `JointNormalizer`
- **Evaluation metrics** (from `eval_ik.py`):
  - **Success rate (SR)**: fraction of 500 random targets where at least 1 of 50 samples achieves position error < 1mm AND orientation error < 1°
  - **Diversity**: mean pairwise joint-space distance among successful samples
  - **Position error** (mm): L2 norm of FK translation difference
  - **Orientation error** (deg): geodesic distance on SO(3)
- **Hardware**: NVIDIA RTX 4090, PyTorch 2.11+cu128
- **Logging**: Weights & Biases, project "DiffusionIK"

## Diffusion Framework

### Forward Process

Standard DDPM forward diffusion:
```
q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
```

### Noise Schedules Tested

- **Cosine** (Nichol & Dhariwal 2021): `alpha_bar(t) = cos((t/T + s)/(1+s) * pi/2)^2`, s=0.008
- **Linear**: `beta_t = linspace(1e-4, 0.02, T)`

### Prediction Targets Tested

- **Epsilon (noise)**: model predicts the added noise
- **x0 (clean data)**: model directly predicts denoised joints
- **v-prediction** (Salimans & Ho 2022): `v = sqrt(alpha_bar) * eps - sqrt(1-alpha_bar) * x0`

### Sampling

DDIM (Song et al. 2020) with 50 steps — deterministic reverse diffusion:
```
x_0_pred = (x_t - sqrt(1-alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
eps_pred = (x_t - sqrt(alpha_bar_t) * x_0_pred) / sqrt(1-alpha_bar_t)
x_{t-1} = sqrt(alpha_bar_{t-1}) * x_0_pred + sqrt(1-alpha_bar_{t-1}) * eps_pred
```

### Classifier-Free Guidance (CFG)

During training, condition is zeroed with probability `cfg_dropout=0.1`. At inference:
```
pred = pred_uncond + (1 + guidance_scale) * (pred_cond - pred_uncond)
```
`guidance_scale = 1.5` was used.

### Gradient Refinement

Post-hoc optimization through differentiable FK:
```python
joints = diffusion_output.clone().requires_grad_(True)
optimizer = Adam([joints], lr=0.005)
for step in range(num_steps):
    R_pred, t_pred = FK(joints)
    loss = MSE(t_pred, t_target) + MSE(R_pred, R_target)
    loss.backward()
    optimizer.step()
```

## Architectures

### MLPDenoiser

Feed-forward MLP with time embedding injected at every layer:
```
input: concat(noisy_joints[7], condition[12]) -> Linear -> hidden
each layer: Linear(hidden, hidden) + LayerNorm + SiLU + time_proj(sinusoidal_emb)
output: Linear(hidden, 7)
```

### ResMLPDenoiser (best architecture)

MLP with residual blocks and per-block time conditioning:
```
input: concat(noisy_joints[7], condition[12]) -> Linear -> hidden
each block:
    h = SiLU(Linear(LayerNorm(x))) + time_proj(sinusoidal_emb)
    x = x + Linear(LayerNorm(h))   # residual connection
output: Linear(LayerNorm(x), 7)
```

### TransformerDenoiser (tested, not selected)

Each joint as a separate token, with time and condition as prepended tokens:
```
joint_tokens: Linear(joint_scalar, d_model) + learned_pos_embed  [B, 7, D]
prepend: [time_token, cond_token, joint_tokens]  [B, 9, D]
N layers of TransformerEncoderLayer (self-attention + FFN)
output: Linear(joint_tokens, 1) per token  [B, 7]
```

### DiT (Diffusion Transformer with AdaLN)

Adaptive Layer Normalization conditioning instead of token prepending:
```
ada_cond = MLP(concat(time_emb, cond_emb))
each block:
    h = AdaLN(x, ada_cond)  # scale/shift from ada_cond
    h = MultiHeadAttention(h, h, h)
    x = x + h
    x = x + FFN(AdaLN(x, ada_cond))
```

## Experiments

### Default Config (unless overridden)

| Parameter | Value |
|---|---|
| T (diffusion steps) | 200 |
| Schedule | cosine |
| Prediction target | epsilon |
| Epochs | 150 |
| Batch size | 4096 |
| Batches per epoch | 300 |
| Total training steps | 45,000 |
| Learning rate | 5e-4 (OneCycleLR, 5% warmup, cosine anneal) |
| Weight decay | 1e-4 |
| Optimizer | AdamW |
| DDIM sample steps | 50 |
| Gradient clip | max_norm=1.0 |
| Eval targets | 500 |
| Samples per target | 50 |

### Phase 1: Architecture & Model Size

| Experiment | Architecture | Hidden | Layers | Params | Pos Error (mm) | SR | Time (s) |
|---|---|---|---|---|---|---|---|
| v01_resmlp_s | ResMLP | 256 | 6 | 1,034,247 | 78.7 | 0.00% | 139 |
| v02_resmlp_m | ResMLP | 512 | 8 | 4,795,143 | 60.4 | 0.00% | 292 |
| **v03_resmlp_l** | **ResMLP** | **1024** | **12** | **26,887,431** | **56.3** | **0.20%** | **1037** |
| v04_mlp_m | MLP | 512 | 8 | 2,684,679 | 73.4 | 0.00% | 181 |

**Finding**: Larger ResMLP is best. Residual connections outperform plain MLP (+17% lower pos error at same width). More parameters help — clear scaling trend from 1M to 27M. First non-zero SR at 27M params.

**Selected**: ResMLP 1024×12 (26.9M params)

*Note*: Transformer (4.8M, 62.2mm, 3875s) and DiT were also tested in an earlier run but were 10-20x slower per epoch due to attention over 9 tokens. Attention is unnecessary overhead for 7D data — the ResMLP processes the full 7D vector at once more efficiently.

### Phase 2: Prediction Target

| Experiment | Prediction | Pos Error (mm) | SR | Time (s) |
|---|---|---|---|---|
| (v03, baseline) | epsilon | 56.3 | 0.20% | 1037 |
| v05_x0pred | x0 | 111.0 | 0.00% | 1037 |
| v06_vpred | v | 57.9 | 0.00% | 1037 |

**Finding**: Epsilon prediction is clearly best. x0 prediction performs 2x worse — the model struggles to predict clean data directly because the target changes drastically with noise level. v-prediction is competitive but slightly worse than epsilon.

**Selected**: epsilon prediction

### Phase 3: Schedule & Diffusion Steps

| Experiment | Schedule | T | Pos Error (mm) | SR | Time (s) |
|---|---|---|---|---|---|
| (v03, baseline) | cosine | 200 | 56.3 | 0.20% | 1037 |
| v07_linear | linear | 200 | 38.1 | 0.00% | 1036 |
| v08_T50 | cosine | 50 | 137.8 | 0.00% | 1034 |

**Finding**: Linear schedule achieves 32% lower position error (38.1mm vs 56.3mm) but 0% SR. T=50 is far too few diffusion steps — model cannot learn the denoising process in so few levels. Cosine schedule selected due to higher SR (0.20% > 0%).

**Selected**: cosine schedule, T=200

*Note*: The linear schedule's lower pos_mm but zero SR suggests it produces tighter but possibly less diverse samples, missing the narrow success threshold. Cosine schedule produces slightly more variance, occasionally hitting the 1mm/1° window.

### Phase 4: Advanced Techniques

| Experiment | Technique | Pos Error (mm) | SR | Time (s) |
|---|---|---|---|---|
| (v03, baseline) | none | 56.3 | 0.20% | 1037 |
| **v09_cfg** | **CFG (gs=1.5, dropout=0.1)** | **16.9** | **2.40%** | **1070** |
| v10_fkloss | FK loss (weight=0.1) | 55.6 | 0.00% | 1070 |
| v11_fkloss_strong | FK loss (weight=1.0) | 78.8 | 0.00% | 1070 |

**Finding**: Classifier-Free Guidance is transformative — 3.3x reduction in position error and 12x improvement in SR. The guidance pushes samples closer to the conditioned target at the cost of some diversity. FK auxiliary loss during training did not help and actually hurt with strong weight (1.0), likely because the x0 prediction at high noise levels is too noisy to produce meaningful FK gradients.

**Selected**: CFG with guidance_scale=1.5, cfg_dropout=0.1

### Phase 5: Final Optimization (Longer Training)

| Experiment | Steps | Batch | LR | Pos Error (mm) | SR | Time (s) |
|---|---|---|---|---|---|---|
| (v09, baseline) | 45K | 4096 | 5e-4 | 16.9 | 2.40% | 1070 |
| v12_long | 200K | 4096 | 1e-3 | 17.6 | 1.60% | 4575 |
| v13_long_hrlr | 200K | 8192 | 3e-3 | 17.9 | 2.20% | 8574 |

**Finding**: Longer training (4.4x more steps) did NOT improve over the shorter run. The model appears to converge within ~45K steps for this problem. Diminishing returns from additional training — the diffusion loss plateaus around 0.19-0.20.

**Selected**: Original 45K-step training is sufficient

### Phase 6: Gradient Refinement at Inference

| Experiment | Refine Steps | Refine LR | Pos Error (mm) | SR | Time (s) |
|---|---|---|---|---|---|
| v14_cfg_baseline | 0 | — | 17.0 | 1.00% | 1067 |
| **v15_refine50** | **50** | **0.01** | **17.4** | **97.20%** | **1127** |
| **v16_refine200** | **200** | **0.005** | **17.0** | **99.40%** | **1304** |
| **v17_refine500** | **500** | **0.003** | **17.3** | **100.00%** | **1656** |

**Finding**: Gradient refinement through differentiable FK is the single most impactful technique. Starting from the diffusion model's ~17mm average output, Adam optimizer rapidly converges through the FK landscape to sub-millimeter accuracy. 50 steps already achieve 97.2% SR; 500 steps reach 100%. The refinement adds minimal overhead (~60s for 50 steps, ~600s for 500 steps over 500 evaluation targets × 50 samples each). Note that `pos_mm` reported is the quick_eval metric on diffusion output (before refinement) — the actual post-refinement position error is sub-1mm for successful samples.

## Final Best Configuration

```python
{
    "arch": "resmlp",
    "hidden": 1024,
    "layers": 12,
    "params": "26,887,431",
    "T": 200,
    "schedule": "cosine",
    "pred_type": "eps",
    "epochs": 150,
    "batch_size": 4096,
    "batches_per_epoch": 300,
    "lr": "5e-4 (OneCycleLR)",
    "cfg_dropout": 0.1,
    "guidance_scale": 1.5,
    "sample_steps": 50,  # DDIM
    "refine_steps": 200,  # post-hoc FK refinement
    "refine_lr": 0.005,
    "success_rate": "99.4%",
}
```

## Key Insights

1. **Diffusion provides diversity, refinement provides precision.** The diffusion model generates diverse initial guesses across the multi-modal IK solution space. Gradient refinement through differentiable FK then pushes each sample to sub-millimeter accuracy. Neither alone is sufficient — diffusion without refinement gives ~2% SR; refinement without diffusion (i.e., from random initialization) would lack the structured starting points.

2. **CFG is essential for conditional generation quality.** Classifier-Free Guidance improved position error by 3.3x (56mm → 17mm). Without it, the model struggles to strongly condition on the target pose.

3. **Residual connections matter more than attention for low-D problems.** ResMLP outperformed both plain MLP and Transformer/DiT. Self-attention over 7 joint tokens adds 10-20x computational overhead with no accuracy benefit. For low-dimensional problems, dense residual MLPs are the sweet spot.

4. **Epsilon prediction is the right parameterization.** x0 prediction failed (2x worse) because the denoising target varies dramatically across noise levels. Epsilon prediction provides a stable learning signal.

5. **More training has diminishing returns.** The model converges around 45K steps (150 epochs × 300 batches). Going to 200K steps did not meaningfully improve quality, suggesting the bottleneck is model capacity or the diffusion framework itself, not optimization.

6. **T=200 is sufficient for 7D data.** T=1000 (standard for images) is wasteful for a 7-dimensional problem. T=50 is too few. T=200 provides adequate noise level granularity.

7. **FK auxiliary loss hurts training.** Adding an FK consistency loss during diffusion training did not help (and hurt at high weight). The noise prediction loss is sufficient — the FK refinement at inference time is a better place to inject physical constraints.

## Comparison with Direct Regression Baseline

The legacy `research_ik_legacy.py` trained a direct regression model (`IKModelCondJ0`, 352K params) that achieved 97.7-98.2% SR by conditioning on j0 and predicting the remaining 6 joints. The diffusion model with refinement achieves 99.4-100% SR. Key differences:

| | Direct Regression (cond_j0) | Diffusion + Refinement |
|---|---|---|
| Params | 352K | 26.9M |
| Training steps | ~100K | 45K |
| SR | 97.7-98.2% | 99.4-100% |
| Diversity | Limited (1 solution per j0) | High (50 diverse solutions from noise) |
| Inference | Single forward pass | 50 DDIM steps + 200 refinement steps |

The diffusion approach trades compute for diversity and slightly higher accuracy. Its main advantage is producing multiple diverse valid solutions from a single target, which is valuable for motion planning and collision avoidance.

## Visualization Results

For the "Forward" pose (target EE: 174.6, 6.7, 805.2 mm), the diffusion model with 200 refinement steps generates 100 diverse solutions:

- Raw diffusion output: mean pos error 25.5mm, 0% SR
- After 50 refine steps: mean pos error 0.024mm, 100% SR, diversity 4.23 rad
- After 200 refine steps: mean pos error 0.001mm, 100% SR, diversity 4.23 rad

The solutions show high diversity in joints 1, 3, 5, 7 (which have redundant DOF) while converging on joints 2, 4, 6 (constrained by the kinematic chain).

## Wandb Project

All experiments: https://wandb.ai/yaroslavhladun-somatic/DiffusionIK

Groups:
- `diffusion_v3` — main 11-experiment research run (phases 1-4 + partial phase 5)
- Refinement experiments — final 4 runs (v14-v17)

## Checkpoints

- `best_v14_cfg_baseline.pt` — Best model weights (ResMLP 1024×12, CFG-trained)
- Various `best_v*.pt` files from each experiment

## Code

- `diffusion_ik.py` — Complete implementation (models, scheduler, training, sampling, evaluation, experiment runner)
- `run_refinement.py` — Standalone refinement experiment script
- `visualize_ik.py` — Per-pose statistics and sample solutions
- `print_ik.py` — Quick table of IK solutions for a fixed target
