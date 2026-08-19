"""1コアあたりの time_marching 性能を測る (機種・実装の比較用)。

使い方: cd <リポジトリ or worktree>; python bench_core.py [--reference]
"""
import argparse
import os
import platform
import sys
import time

sys.path.insert(0, 'tests')
import numpy as np

import S2MFD
from conftest import make_cfg, make_grid

p = argparse.ArgumentParser()
p.add_argument('--reference', action='store_true',
               help='参照実装 (融合前) を測る')
p.add_argument('--steps', type=int, default=400)
args = p.parse_args()

cfg = make_cfg('parameters/inference.py', ix=128, jx=128)
grid = make_grid(cfg)
setup = S2MFD.Setup(cfg, grid)

if args.reference and hasattr(S2MFD.physics, 'time_marching_reference'):
    march = S2MFD.physics.time_marching_reference
    label = 'reference (融合前)'
else:
    march = S2MFD.physics.time_marching
    label = 'fused (融合後)' if hasattr(S2MFD.physics, 'time_marching_reference') \
        else 'current (main)'

rng = np.random.default_rng(0)
Bph = rng.standard_normal((grid.ixg, grid.jxg)) * 0.1
Aph = rng.standard_normal((grid.ixg, grid.jxg)) * 1e7
dt = 1.0e5

# ウォームアップ (JIT コンパイル)
for _ in range(5):
    Bph2, Aph2 = march(Bph, Aph, dt, cfg, grid, setup)

t0 = time.perf_counter()
for _ in range(args.steps):
    Bph2, Aph2 = march(Bph, Aph, dt, cfg, grid, setup)
elapsed = time.perf_counter() - t0

per_call_ms = elapsed / args.steps * 1000
# 1個体 = 約137年 / dt≈3.4日 ≈ 14700 ステップ × 2 substep
substeps = 14700 * 2
print(f'{platform.node():10} {label:20} '
      f'{per_call_ms:7.3f} ms/substep  '
      f'→ 1個体 {per_call_ms*substeps/1000/60:6.2f} 分')
