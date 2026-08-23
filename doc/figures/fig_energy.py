"""エネルギー収支の検証 (Rempel 2006 図 7 と表 1 に対応).

(a) エネルギーの流れの模式図に**実測値**を書き込む。
(b) 収支式 (20)-(22) の残差が解像度とともに消えること。
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, 'run_paris')]
import style                                            # noqa: E402
import convergence                                      # noqa: E402

L_SUN = 3.828e33
#: Rempel (2006) 表 1 列 5 (alpha_0 = 0.25 m/s)。全 8 項が揃う列。
PAPER = dict(QL=0.011, Qnu=0.405, QC=0.445, QLO=0.149, QB=0.414,
             QnuM=0.007, QLM=0.024, Qeta=0.172)


def _box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                boxstyle='round,pad=0.012,rounding_size=0.02',
                                fc=fc, ec=style.INK3, lw=1.0, zorder=2))
    ax.text(x, y, text, ha='center', va='center', fontsize=10,
            color=style.INK, zorder=3, linespacing=1.5)


def _arrow(ax, p0, p1, label, sub=None):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=15,
                                 color=style.INK2, lw=1.6, zorder=4,
                                 shrinkA=2, shrinkB=2))
    mx, my = 0.5*(p0[0] + p1[0]), 0.5*(p0[1] + p1[1])
    ax.text(mx, my, label, ha='center', va='center', fontsize=9,
            color=style.INK, zorder=5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.22', fc=style.SURFACE,
                      ec='none'))
    if sub:
        ax.text(mx, my - 0.052, sub, ha='center', va='center', fontsize=8,
                color=style.INK2, zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', fc=style.SURFACE,
                          ec='none'))


def _flow(ax, meas):
    """Rempel (2006) 図 7 に実測値を書き込んだ模式図."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    pale, mid, warm = '#e8f1fd', '#cde2fb', '#fbe3d6'
    _box(ax, 0.5, 0.925, 0.68, 0.10, 'internal energy', pale)
    _box(ax, 0.24, 0.58, 0.36, 0.13, 'differential\nrotation', mid)
    _box(ax, 0.79, 0.58, 0.33, 0.13, 'meridional\nflow', mid)
    _box(ax, 0.5, 0.10, 0.68, 0.10, 'toroidal magnetic field', warm)

    def v(key, fmt='{:.3f}'):
        return (fmt.format(meas[key]) + '   ('
                + fmt.format(PAPER[key]) + ')')

    # 内部エネルギー <-> 差動回転 / 子午面流
    _arrow(ax, (0.13, 0.875), (0.13, 0.645), r'$Q_\Lambda$',
           f'{meas["QL"]:.4f} $F_\\odot$   ({PAPER["QL"]:.3f})')
    _arrow(ax, (0.35, 0.645), (0.35, 0.875), r'$Q_\nu^\Omega$', v('Qnu'))
    _arrow(ax, (0.66, 0.645), (0.66, 0.875), r'$Q_\nu^M$',
           v('QnuM', '{:.3f}'))
    _arrow(ax, (0.93, 0.645), (0.93, 0.875), r'$Q_B$', v('QB'))
    # 差動回転 -> 子午面流
    _arrow(ax, (0.43, 0.58), (0.62, 0.58), r'$Q_C$', v('QC'))
    # 磁場へ
    _arrow(ax, (0.24, 0.515), (0.24, 0.155), r'$Q_L^\Omega$', v('QLO'))
    _arrow(ax, (0.79, 0.515), (0.79, 0.155), r'$Q_L^M$', v('QLM'))
    _arrow(ax, (0.5, 0.155), (0.5, 0.36), r'$Q_\eta$', v('Qeta'))
    ax.text(0.5, 0.44, 'to internal energy', ha='center', va='center',
            fontsize=8, color=style.INK3)
    ax.set_title('(a) Energy flow with measured values  '
                 '($108\\times72$, $\\alpha_0$ = 0.25 m s$^{-1}$)\n'
                 'this work   (Rempel 2006, Table 1 col. 5);  '
                 'ratios to $Q_\\Lambda$ unless noted',
                 fontsize=11, fontweight='bold')


def main(outdir):
    style.apply()
    os.chdir(ROOT)

    # --- 実測値 (ダイナモの完走ラン) --------------------------------------
    import table1
    r = table1.analyse('v2_a250')
    meas = dict(QL=r['QL_Fsun'], Qnu=r['Qnu'], QC=r['QC'], QLO=r['QLO'],
                QB=r['QB'], QnuM=r['QnuM'], QLM=r['QLM'], Qeta=r['Qeta'])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0),
                             gridspec_kw=dict(width_ratios=[1.15, 1]))
    _flow(axes[0], meas)

    # --- (b) 収支式の残差 vs 解像度 --------------------------------------
    ax = axes[1]
    tab = convergence.collect()
    ns, r_om, r_m = [], [], []
    for (nx, ny, cs), r in sorted(tab.items()):
        if abs(cs - 0.30) > 1e-12 or not r['sat']:
            continue
        hh = np.load(f"results_rempel/{r['tag']}/history.npz")['hist']
        kk = max(1, len(hh)//10)
        q = hh[-kk:, 11].mean()
        ns.append(nx)
        r_om.append(abs((hh[-kk:, 11] - hh[-kk:, 12] - hh[-kk:, 13]).mean()/q))
        r_m.append(abs((hh[-kk:, 13] - hh[-kk:, 14] - hh[-kk:, 15]).mean()/q))
    ns = np.array(ns, float)
    ax.plot(ns, r_om, 'o-', color=style.BLUE[450],
            markeredgecolor=style.SURFACE, markeredgewidth=1.2,
            label=r'eq. (20)  $Q_\Lambda-Q_\nu^\Omega-Q_C$')
    ax.plot(ns, r_m, 's-', color=style.ORANGE_RAMP[2],
            markeredgecolor=style.SURFACE, markeredgewidth=1.2,
            label=r'eq. (21)  $Q_C-Q_\nu^M-Q_B$')
    ax.axhline(0.001, color=style.REFERENCE, lw=1.4, ls='--')
    ax.annotate('accuracy stated in Rempel (2006), Table 1 note: 0.001',
                xy=(0.98, 0.001), xycoords=('axes fraction', 'data'),
                xytext=(0, 5), textcoords='offset points', ha='right',
                fontsize=8.5, color=style.REFERENCE)
    ref = r_m[0]*(ns[0]/ns)**2
    ax.plot(ns, ref, color=style.INK3, lw=1.2, ls=':', label='$N_r^{-2}$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.xaxis.set_major_locator(FixedLocator(list(ns)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}'))
    ax.set_xlabel('$N_r$')
    ax.set_ylabel(r'residual  /  $Q_\Lambda$')
    ax.set_title('(b) Energy-budget residual vs resolution\n'
                 'the paper grid $N_r=108$ is 10x worse than its own accuracy',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', frameon=True, facecolor=style.SURFACE,
              edgecolor='none', framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(outdir, 'energy_budget.png')
    fig.savefig(out)
    print('wrote', out)


if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'out')
    os.makedirs(d, exist_ok=True)
    main(d)
