#!/usr/bin/env python3
"""Train ONLY the paper's final diffusion configuration (v14_cfg_baseline):
ResMLP 1024x12, eps-prediction, cosine T=200, CFG (w=1.5, p_drop=0.1),
45,000 steps — and export an fp16 copy of the best checkpoint.

Works offline (wandb disabled automatically if no login). Suitable for
Kaggle/Colab T4: ~1.5-3 h.

    python train_v14.py            # full training
    python train_v14.py --smoke    # 2-epoch wiring test
"""
import argparse
import os

os.environ.setdefault('WANDB_MODE', 'disabled')

import torch

from diffusion_ik import base, run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--tag', default='v14_cfg_baseline')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    cfg = base(
        tag=args.tag, arch='resmlp', hidden=1024, layers=12,
        cfg_dropout=0.1, guidance_scale=1.5,
        notes='paper final config (retrain for revision)')
    if args.smoke:
        cfg.update(epochs=2, batches_per_epoch=50, batch_size=256,
                   qeval_every=1, feval_every=2, eval_targets=8)
    run_experiment(cfg, device)

    best = f'best_{args.tag}.pt'
    if os.path.exists(best):
        sd = torch.load(best, map_location='cpu')
        torch.save({k: (v.half() if v.is_floating_point() else v)
                    for k, v in sd.items()}, f'best_{args.tag}_fp16.pt')
        print(f'saved best_{args.tag}_fp16.pt '
              f'({os.path.getsize(f"best_{args.tag}_fp16.pt")/1e6:.1f} MB)')
    else:
        print(f'WARNING: {best} was not produced (SR never improved above 0); '
              f'check the training log')


if __name__ == '__main__':
    main()
