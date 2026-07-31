#!/usr/bin/env python3
"""Export the unified benchmark pose sets (and FK check vectors) for kdl_baseline.py."""
import json

import torch

from fk import FK
from benchmark_block_a import make_test_set, make_near_singular_set

device = torch.device('cpu')
fk = FK(device)

q_m, R_m, t_m = make_test_set(500, 0, device)
q_s, R_s, t_s, _ = make_near_singular_set(500, 0, device)


def pack(q, R, t):
    return [{'q': q[i].tolist(),
             'R': R[i].tolist(),
             't': t[i].squeeze(-1).tolist()} for i in range(q.shape[0])]


torch.manual_seed(7)
q_chk = torch.rand(50, 7) * 2 - 1
data = {
    'fk_check': pack(q_chk, *fk(q_chk)),
    'main': pack(q_m, R_m, t_m),
    'singular': pack(q_s, R_s, t_s),
}
json.dump(data, open('poses_export.json', 'w'))
print('exported', {k: len(v) for k, v in data.items()})
