#!/usr/bin/env python3
"""One-shot GPU runner for the remaining Block A benchmark rows.

Auto-detects which checkpoints are present in the repo and measures every
row they enable — main + near-singular set, raw + FK refinement — appending
to results_block_a/results.json (existing rows are kept):

    checkpoint                      rows
    best_v14_cfg_baseline_fp16.pt   diffusion[-refine][-singular]
    flow_xarm7.pt                   flow[-refine][-singular]

Usage (Kaggle T4 / any CUDA box):
    python run_bench_gpu.py [--flow-temp T]

--flow-temp: sampling temperature for the flow rows; pick it first with
    python flow_baseline.py sweep --ckpt flow_xarm7.pt
(selected on a validation pose set, seed=1, so the test set stays clean).

Safe to re-run: each invocation just re-measures the rows for the
checkpoints it finds. ~10-15 min per checkpoint on a T4.
"""
import argparse
import os
import subprocess
import sys

import torch

ROWS = [
    ('best_v14_cfg_baseline_fp16.pt', '--ckpt-diffusion',
     'diffusion,diffusion-refine,diffusion-singular,diffusion-refine-singular'),
    ('flow_xarm7.pt', '--ckpt-flow',
     'flow,flow-refine,flow-singular,flow-refine-singular'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flow-temp', type=float, default=1.0)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit('CUDA GPU required — this runner exists so the heavy rows '
                 'are timed on GPU (the protocol notes learned methods are '
                 'GPU-timed). Use Kaggle T4 x2.')
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    ran = False
    for ckpt, flag, methods in ROWS:
        if not os.path.exists(ckpt):
            print(f'-- {ckpt} not found, skipping: {methods}', flush=True)
            continue
        cmd = [sys.executable, 'benchmark_block_a.py',
               '--methods', methods, flag, ckpt,
               '--targets', '500', '--samples', '50']
        if flag == '--ckpt-flow':
            cmd += ['--flow-temp', str(args.flow_temp)]
        print('>>>', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        ran = True
    if ran:
        print('\n===== results_block_a/table.md =====')
        print(open('results_block_a/table.md').read())
    else:
        print('No checkpoints found — nothing to run.')


if __name__ == '__main__':
    main()
