"""緩和済みの状態を別解像度の格子へ双線形内挿する。

摂動量 (om1, ro1, se1) と速度は同じ背景成層に対する量なので、
そのまま内挿してよい。ゴーストセルは呼び出し側の境界条件で埋め直す。
"""
import sys, numpy as np
sys.path.insert(0, 'tests')
import os
import S2MFD
PARFILE=os.environ.get('S2MFD_PARFILE','parameters/rempel06.py')
from conftest import make_cfg, make_grid

src_file, nx, ny, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
d = np.load(src_file)
# 元の格子座標はファイルに入っているのでそれを使う (margin の推測を避ける)
rs, ts = d['rr'], d['th']
# grid_stretch はパラメタファイルの値をそのまま使う。
# 以前は既定 0.0 で上書きしていたため、rempel06_paper.py の 2 点集中格子
# ([2.0, 3.0]) が一様格子に化けていた。
over = {}
if len(sys.argv) > 5:
    over['grid_stretch'] = float(sys.argv[5])
cfg_t = make_cfg(PARFILE, ix=nx, jx=ny, **over)
print(f'  target: {PARFILE}  rrmax={cfg_t.rrmax/cfg_t.RSUN:.3f}R  stretch={cfg_t.grid_stretch}')
gt = make_grid(cfg_t)
print(f"{len(rs)}x{len(ts)} -> {gt.ixg}x{gt.jxg}")

def interp2d(q, rs, ts, rt, tt):
    # r 方向 -> theta 方向の順に 1 次元内挿
    tmp = np.empty((len(rt), q.shape[1]))
    for j in range(q.shape[1]):
        tmp[:, j] = np.interp(rt, rs, q[:, j])
    outq = np.empty((len(rt), len(tt)))
    for i in range(len(rt)):
        outq[i, :] = np.interp(tt, ts, tmp[i, :])
    return np.ascontiguousarray(outq)

res = {}
for k in ('om1', 'vrr', 'vth', 'ro1', 'se1'):
    res[k] = interp2d(d[k], rs, ts, gt.rr, gt.th)
    print(f"  {k}: |max| {np.abs(d[k]).max():.4e} -> {np.abs(res[k]).max():.4e}")
res['t'] = d['t']
np.savez(out, **res)
print('wrote', out)
