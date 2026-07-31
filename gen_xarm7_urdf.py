#!/usr/bin/env python3
"""Generate an xArm7 URDF that reproduces fk.py EXACTLY.

fk.py composes, per joint i:  T_static_i (fixed xyz+rpy)  ∘  Rz(q_i).
The equivalent URDF chain is 7 revolute joints about local Z, each with
<origin xyz rpy> equal to the static transform of fk.py, plus joint limits
from common.py. Using this generated file (instead of the vendor URDF)
guarantees that every solver in the benchmark works on the *identical*
kinematic model used to train and evaluate DiffusionIK.
"""
from math import pi

from common import JOINTS_LOWER_LIMIT, JOINTS_UPPER_LIMIT

# xyz (m) + rpy (rad) static transforms, verbatim from fk.py
STATIC = [
    (0, 0, 0.267, 0, 0, 0),
    (0, 0, 0, -1.5708, 0, 0),
    (0, -0.293, 0, 1.5708, 0, 0),
    (0.0525, 0, 0, 1.5708, 0, 0),
    (0.0775, -0.3425, 0, 1.5708, 0, 0),
    (0, 0, 0, 1.5708, 0, 0),
    (0.076, 0.097, 0, -1.5708, 0, 0),
]


def build():
    parts = ['<?xml version="1.0"?>', '<robot name="xarm7_from_fk">',
             '  <link name="link0"/>']
    for i, (x, y, z, r, p, yaw) in enumerate(STATIC, start=1):
        parts += [
            f'  <link name="link{i}"/>',
            f'  <joint name="joint{i}" type="revolute">',
            f'    <parent link="link{i-1}"/>',
            f'    <child link="link{i}"/>',
            f'    <origin xyz="{x} {y} {z}" rpy="{r} {p} {yaw}"/>',
            '    <axis xyz="0 0 1"/>',
            f'    <limit lower="{JOINTS_LOWER_LIMIT[i-1]:.10f}" '
            f'upper="{JOINTS_UPPER_LIMIT[i-1]:.10f}" effort="100" velocity="3.14"/>',
            '  </joint>',
        ]
    parts.append('</robot>')
    return '\n'.join(parts) + '\n'


if __name__ == '__main__':
    urdf = build()
    with open('xarm7_from_fk.urdf', 'w') as f:
        f.write(urdf)
    print('wrote xarm7_from_fk.urdf')
    print(urdf[:400])
