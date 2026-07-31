#!/usr/bin/env python3
"""
kdl_baseline.py — Orocos KDL (LMA) IK baseline for the unified xArm7 benchmark.

RUN UNDER /usr/bin/python3.12 (PyKDL from the system package is built for it):
    python3 kdl_export_poses.py            # torch env: dumps poses_export.json
    /usr/bin/python3.12 kdl_baseline.py    # this script: writes kdl_results.json
    python3 kdl_merge.py                   # torch env: merges into results_block_a

The kinematic chain is built to reproduce fk.py exactly:
fk.py composes  F_i (static xyz+rpy)  then  Rz(q_i)  per joint, so the KDL
chain starts with a fixed segment carrying F_1 and alternates
[RotZ joint, tip transform F_{i+1}] afterwards. FK equivalence is asserted
against values exported from the torch implementation before any IK runs.
"""
import json
import math
import random
import time

import PyKDL as kdl

STATIC = [
    (0, 0, 0.267, 0, 0, 0),
    (0, 0, 0, -1.5708, 0, 0),
    (0, -0.293, 0, 1.5708, 0, 0),
    (0.0525, 0, 0, 1.5708, 0, 0),
    (0.0775, -0.3425, 0, 1.5708, 0, 0),
    (0, 0, 0, 1.5708, 0, 0),
    (0.076, 0.097, 0, -1.5708, 0, 0),
]
LO = [-2 * math.pi, -2.059, -2 * math.pi, -0.19198, -2 * math.pi, -1.69297, -2 * math.pi]
HI = [2 * math.pi, 2.0944, 2 * math.pi, 3.927, 2 * math.pi, math.pi, 2 * math.pi]

POS_THR = 1e-3
ORI_THR = 1.0 * math.pi / 180


def frame(x, y, z, r, p, yw):
    return kdl.Frame(kdl.Rotation.RPY(r, p, yw), kdl.Vector(x, y, z))


def build_chain():
    ch = kdl.Chain()
    # fixed segment carrying F_1
    ch.addSegment(kdl.Segment(kdl.Joint(), frame(*STATIC[0])))
    for i in range(7):
        tip = frame(*STATIC[i + 1]) if i + 1 < 7 else kdl.Frame.Identity()
        ch.addSegment(kdl.Segment(kdl.Joint(kdl.Joint.RotZ), tip))
    return ch


def jnt(vals):
    a = kdl.JntArray(len(vals))
    for i, v in enumerate(vals):
        a[i] = v
    return a


def frame_to_Rt(f):
    R = [[f.M[i, j] for j in range(3)] for i in range(3)]
    t = [f.p[i] for i in range(3)]
    return R, t


def pose_err(f, R_t, t_t):
    dp = math.sqrt(sum((f.p[i] - t_t[i]) ** 2 for i in range(3)))
    tr = sum(sum(f.M[i, k] * R_t[i][k] for k in range(3)) for i in range(3))
    c = max(-1.0, min(1.0, (tr - 1) / 2))
    return dp, math.acos(c)


def main():
    data = json.load(open('poses_export.json'))
    chain = build_chain()
    fk = kdl.ChainFkSolverPos_recursive(chain)

    # FK equivalence check against the torch implementation
    max_dp = max_dr = 0.0
    for rec in data['fk_check']:
        f = kdl.Frame()
        fk.JntToCart(jnt(rec['q']), f)
        dp, dr = pose_err(f, rec['R'], rec['t'])
        max_dp, max_dr = max(max_dp, dp), max(max_dr, dr)
    # the torch FK runs in float32; its rounding (~1e-6 per matrix element)
    # is amplified by acos near the identity, so ~1e-3 rad is pure precision
    # noise, 30x below the 1-degree success threshold
    assert max_dp < 1e-5 and max_dr < 2e-3, (max_dp, max_dr)
    print(f'FK equivalence vs fk.py: max pos diff {max_dp:.2e} m, '
          f'max ori diff {max_dr:.2e} rad')

    ik = kdl.ChainIkSolverPos_LMA(chain, 1e-10, 1000)
    rng = random.Random(12345)
    out = {}
    for set_name in ('main', 'singular'):
        poses = data[set_name]
        n = len(poses)
        solved = 0
        errs_p, errs_o, lats, restarts_used = [], [], [], []
        perpose = []
        for rec in poses:
            R_t, t_t = rec['R'], rec['t']
            target = kdl.Frame(
                kdl.Rotation(*[R_t[i][j] for i in range(3) for j in range(3)]),
                kdl.Vector(*t_t))
            best = (float('inf'), float('inf'))
            ok = False
            t0 = time.perf_counter()
            used = 0
            for r in range(10):
                used = r + 1
                q0 = jnt([rng.uniform(LO[i], HI[i]) for i in range(7)])
                q_out = kdl.JntArray(7)
                ik.CartToJnt(q0, target, q_out)
                # joint limits are not enforced by LMA: reject violations
                within = all(LO[i] - 1e-9 <= q_out[i] <= HI[i] + 1e-9
                             for i in range(7))
                f = kdl.Frame()
                fk.JntToCart(q_out, f)
                dp, dr = pose_err(f, R_t, t_t)
                if within and dp + dr < best[0] + best[1]:
                    best = (dp, dr)
                if within and dp < POS_THR and dr < ORI_THR:
                    ok = True
                    break
            lats.append((time.perf_counter() - t0) * 1000)
            restarts_used.append(used - 1)
            if ok:
                solved += 1
            if best[0] != float('inf'):
                errs_p.append(best[0])
                errs_o.append(best[1])
            perpose.append({'solved': ok,
                            'pos_m': None if best[0] == float('inf') else best[0],
                            'ori_rad': None if best[1] == float('inf') else best[1],
                            'lat_ms': lats[-1], 'restarts': used - 1})
        errs_p.sort(); errs_o.sort(); lats_sorted = sorted(lats)
        def q95(a): return a[min(len(a) - 1, int(0.95 * len(a)))] if a else None
        out[set_name] = {
            'n': n, 'solved': solved, 'SR_pct': 100 * solved / n,
            'pos_mm': {'mean': 1000 * sum(errs_p) / len(errs_p),
                       'median': 1000 * errs_p[len(errs_p) // 2],
                       'p95': 1000 * q95(errs_p)},
            'ori_deg': {'mean': math.degrees(sum(errs_o) / len(errs_o)),
                        'median': math.degrees(errs_o[len(errs_o) // 2]),
                        'p95': math.degrees(q95(errs_o))},
            'latency_ms': {'median': lats_sorted[len(lats) // 2],
                           'mean': sum(lats) / len(lats),
                           'p95': q95(lats_sorted)},
            'restarts_mean': sum(restarts_used) / len(restarts_used),
            'perpose': perpose,
        }
        print(set_name, json.dumps({k: v for k, v in out[set_name].items()
                                    if k != 'perpose'})[:220])
    json.dump(out, open('kdl_results.json', 'w'), indent=2)
    print('saved kdl_results.json')


if __name__ == '__main__':
    main()
