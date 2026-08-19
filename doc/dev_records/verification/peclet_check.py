"""GA の探索範囲が格子ペクレ数 Pe > 2 の領域に入るかを定量評価する。

背景: 移流項は純中心差分 (physics_core.advection) で離散化されており、
格子ペクレ数 Pe = |u| Δ / η が 2 を超えると格子スケールの数値振動が生じうる
(コードレビュー 2026-08-19_code_review.md §1.5)。GA は u0 を探索するため、
「探索範囲のどこまでが安全か」を数値で押さえておく必要がある。

使い方:
    python doc/dev_records/verification/peclet_check.py
"""
import numpy as np

import S2MFD
from S2MFD.inference.problem import U0_SPEC


def peclet_max(cfg, grid, uu0):
    """指定した u0 での領域内の最大格子ペクレ数を返す。"""
    cfg.uu0 = float(uu0)
    setup = S2MFD.Setup(cfg, grid)
    m = grid.margin
    speed = np.sqrt(setup.urr[m:-m, m:-m]**2 + setup.uth[m:-m, m:-m]**2)
    et = setup.et[m:-m, m:-m]
    cell = np.minimum(grid.drr, grid.rr[m:-m, None] * grid.dth)
    return float((speed * cell / et).max())


def main():
    cfg = S2MFD.Cfg('parameters/inference.py')
    grid = S2MFD.Grid(ix=cfg.ix, jx=cfg.jx, margin=cfg.margin,
                      rrmin=cfg.rrmin, rrmax=cfg.rrmax,
                      thmin=cfg.thmin, thmax=cfg.thmax)
    print(f'格子: {cfg.ix}x{cfg.jx}, drr = {grid.drr:.3e} cm '
          f'({grid.drr/cfg.RSUN:.5f} R_sun)')
    print(f'セル異方性 r*dth/drr (r=R): {cfg.RSUN*grid.dth/grid.drr:.1f}')
    print()

    print('u0 と最大格子ペクレ数の関係:')
    for uu0 in (500, 700, 900, 1100, 1350):
        pe = peclet_max(cfg, grid, uu0)
        flag = '← Pe>2' if pe > 2 else ''
        print(f'  u0 = {uu0:5.0f} cm/s  →  Pe_max = {pe:.2f} {flag}')

    # Pe は u0 に比例するので、Pe=2 に対応する u0 を線形に逆算する
    pe_ref = peclet_max(cfg, grid, 900.0)
    u_crit = 900.0 * 2.0 / pe_ref
    print(f'\nPe = 2 に相当する u0: {u_crit:.0f} cm/s')

    # 初期集団の事前分布から、実際に到達する max u0(t) の分布を求める
    rng = np.random.default_rng(0)
    n_sample = 200_000
    span = 52.05 * 365 * 86400  # 5周期 (論文の代表的な推定窓) 相当
    t = np.linspace(0, span, 300)
    omega = 2 * np.pi / (span * 2)
    p = {k: rng.uniform(lo, hi, n_sample) for k, (lo, hi) in U0_SPEC.items()}
    lin = (p['as_u'][:, None] * (span - t) + p['ae_u'][:, None] * t) / span
    u0t = (lin
           + p['a1_u'][:, None] * np.sin(1.0 * omega * t)
           + p['a2_u'][:, None] * np.sin(2.0 * omega * t)
           + p['a4_u'][:, None] * np.sin(4.0 * omega * t))
    umax = u0t.max(axis=1)

    print(f'\n初期集団 ({n_sample} サンプル) の max u0(t):')
    print(f'  中央値 {np.median(umax):.0f}, 95%点 {np.percentile(umax, 95):.0f}, '
          f'最大 {umax.max():.0f} cm/s')
    print(f'  Pe>2   となる個体: {100*np.mean(umax > u_crit):.2f}%')
    print(f'  Pe>2.5 となる個体: {100*np.mean(umax > u_crit*1.25):.4f}%')
    print(f'  到達しうる最大 Pe: {pe_ref/900*umax.max():.2f}')
    print('\n解釈: 探索範囲の大半 (>97%) は Pe<2 の安全域にある。'
          '\n      少数の個体が一時的に Pe≈2.0-2.4 の弱い振動域に入りうるが、'
          '\n      発散域 (Pe>>2) には到達しない。')


if __name__ == '__main__':
    main()
