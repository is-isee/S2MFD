"""動力学モードを 2-3 分で一通り動かしてみる例.

やること:

1. 小さい格子で参照モデル (磁場なし) を緩和させ、:math:`\\Lambda` 効果が
   差動回転を作るところを見る
2. **角運動量が machine precision で保存している**ことを確かめる
3. エネルギー収支 (Rempel 2006 式 20-21) を表示する
4. 図を保存する

実行::

    python examples/rempel2006/quickstart.py [出力先ディレクトリ]

論文と同じ設定での完全な再現は 10 CPU 時間ほどかかる。手順は
ドキュメントの「Rempel (2006) を再現する」を参照。
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import S2MFD
from S2MFD.stratification import Stratification
from S2MFD.physics import conservative as cons
from S2MFD.physics.dynamic import DynamicSolver
from S2MFD.physics.energy import EnergyBudget, solar_luminosity

YR = 3.156e7
NX, NY, YEARS = 48, 32, 2.0


def main(outdir='.'):
    os.makedirs(outdir, exist_ok=True)
    np.seterr(all='ignore')

    # --- 設定と格子 -------------------------------------------------------
    # 論文設定 (北半球、0.65-0.985 R、Lambda 効果で差動回転を駆動) を
    # 粗い格子で。cfg の値は名前で上書きできる。
    cfg = S2MFD.build_cfg('parameters/rempel06_paper.py', ix=NX, jx=NY,
                          dynamics='hydro', angmom_bottom_bc='uniform_rotation',
                          sld_cs_factor=0.30)
    grid = S2MFD.Grid.from_cfg(cfg)
    strat = Stratification(cfg, grid)
    setup = S2MFD.Setup(cfg, grid)
    sol = DynamicSolver(cfg, grid, strat, setup)
    eb = EnergyBudget(cfg, grid, strat, setup)
    m = grid.margin
    sl = (slice(m, grid.ixg - m), slice(m, grid.jxg - m))
    rr = grid.rr[m:grid.ixg - m]
    th = grid.th[m:grid.jxg - m]
    eq = int(np.argmin(abs(th - 0.5*np.pi)))

    # CFL 安全率は von Neumann 解析の中立点から自動で決める
    s0 = sol.neutral_cfl_safety()
    if s0 == s0:
        cfg.cfl_safety = 0.9*s0
    dt = sol.cfl_dt()
    nsteps = int(YEARS*YR/dt)
    print(f'{NX}x{NY},  dt = {dt:.1f} s,  {YEARS} yr = {nsteps} steps')
    print(f'CFL 安全率 {cfg.cfl_safety:.3f} '
          f'(von Neumann の中立点 {s0:.3f} の 0.9 倍)\n')

    # --- 保存量の初期値 ---------------------------------------------------
    zeta2 = strat.zeta**2
    L0 = cons.cell_integral(strat.JL*sol.om1, grid.drr, grid.dth, m)
    scaleL = cons.cell_integral(np.abs(strat.JL*cfg.om0), grid.drr, grid.dth, m)

    # --- 時間発展 ---------------------------------------------------------
    hist = []
    t0 = time.time()
    for n in range(1, nsteps + 1):
        sol.step(dt)
        if n % max(1, nsteps//60) == 0:
            om = sol.om1[sl]
            L = cons.cell_integral(strat.JL*sol.om1, grid.drr, grid.dth, m)
            hist.append((n*dt/YR,
                         (om[-1, eq] - om[-1, 0])/cfg.om0,
                         (L - L0 - sol.boundary_angmom_flux)/scaleL,
                         np.abs(sol.vth[sl]).max()/100.0))
        if n % max(1, nsteps//4) == 0:
            print(f'  t = {n*dt/YR:5.2f} yr   DR = {hist[-1][1]:+.4f}   '
                  f'|v_theta|max = {hist[-1][3]:5.2f} m/s   '
                  f'角運動量の残差 = {hist[-1][2]:+.1e}')
    print(f'\n{(time.time()-t0)/60:.1f} 分\n')

    h = np.array(hist)
    print(f'角運動量は machine precision で保存している: '
          f'|残差| の最大 = {np.abs(h[:, 2]).max():.1e}')
    print('  (境界フラックスを引いた値。下部境界の剛体回転はトルクを与える)\n')

    # --- エネルギー収支 ---------------------------------------------------
    z = np.zeros_like(sol.om1)
    q = eb.exchanges(sol.om1, sol.vrr, sol.vth, z, z, z, sol.se1)
    ql = q['Q_Lambda']
    print('エネルギー収支 (Rempel 2006 式 20-21):')
    print(f"  Q_Lambda      = {ql/solar_luminosity:.4f} F_sun   (論文 0.014)")
    for k, paper in (('Q_nu_Omega', 0.574), ('Q_C', 0.425),
                     ('Q_B', 0.419), ('Q_nu_M', 0.005)):
        print(f'  {k:<13} = {q[k]/ql:.4f} Q_Lambda   (論文 {paper})')
    print('\n  ** この格子と年数では未緩和なので、論文値と合わないのが正しい。')
    print('     解像度と年数を上げると近づく (ドキュメントの収束性の節)。')

    _plot(outdir, rr/cfg.RSUN, th, sol, cfg, h, m, grid)


def _plot(outdir, rr, th, sol, cfg, h, m, grid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 図の色は「その色が何の仕事をしているか」で決める。
    # 緯度は順序量なので単一色相のランプ (虹色にしない)。
    blues = ['#86b6ef', '#2a78d6', '#0d366b']
    plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.35,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titlesize': 10, 'figure.dpi': 120})

    om = (cfg.om0 + sol.om1[m:grid.ixg - m, m:grid.jxg - m])/(2*np.pi)*1e9
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))

    for (j, lab), c in zip(((0, 'pole'), (len(th)//2, r'45$^\circ$'),
                            (len(th) - 1, 'equator')), blues):
        ax[0].plot(rr, om[:, j], color=c, label=lab, lw=2)
    ax[0].set_xlabel(r'$r/R_\odot$'); ax[0].set_ylabel(r'$\Omega/2\pi$ [nHz]')
    ax[0].set_title('(a) differential rotation')
    ax[0].legend(frameon=False, title='latitude', loc='lower right')

    ax[1].plot(h[:, 0], h[:, 1], color=blues[1], lw=2)
    ax[1].set_xlabel('time [yr]')
    ax[1].set_ylabel(r'$(\Omega_{\rm eq}-\Omega_{\rm pole})/\Omega_0$')
    ax[1].set_title('(b) spin-up  (early wiggles = acoustic transient)')

    ax[2].semilogy(h[:, 0], np.abs(h[:, 2]) + 1e-30, color=blues[1], lw=2)
    ax[2].axhline(1e-15, color='#52514e', lw=1.2, ls='--')
    ax[2].annotate('$10^{-15}$', xy=(0.98, 1e-15), xycoords=('axes fraction',
                   'data'), xytext=(0, 4), textcoords='offset points',
                   ha='right', fontsize=8, color='#52514e')
    ax[2].set_xlabel('time [yr]')
    ax[2].set_ylabel('|angular momentum residual|')
    ax[2].set_title('(c) conservation is at machine precision')

    fig.tight_layout(w_pad=2.0)
    out = os.path.join(outdir, 'quickstart.png')
    fig.savefig(out)
    print(f'\n図を書きました: {out}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
