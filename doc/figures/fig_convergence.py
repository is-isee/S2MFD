"""収束性の図: 解像度と人工拡散の二重極限.

このプロジェクトの中心的な主張。cs (人工拡散の強さ) は**順序量**なので
categorical ではなく青の単一色ランプで塗る。
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, 'run_paris')]
import style                                            # noqa: E402
import convergence                                      # noqa: E402

PAPER_DR = 0.27


def _fit(n, y):
    from scipy.optimize import least_squares

    def res(q):
        return q[0] + q[1]*n**(-q[2]) - y

    r = least_squares(res, [0.27, 100.0, 2.0],
                      bounds=([-1, -1e6, 0.5], [1, 1e6, 6]))
    return r.x[0], r.x[2], float(np.sqrt(np.mean(r.fun**2)))


def main(outdir):
    style.apply()
    os.chdir(ROOT)
    tab = convergence.collect()

    series = {}
    for (nx, ny, cs), r in tab.items():
        if r['sat']:
            series.setdefault(cs, []).append((nx, r['DR']))
    css = sorted(series, reverse=True)
    colors = dict(zip(css, style.ordinal_blues(len(css))))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # --- (a) DR vs 1/N^2 -------------------------------------------------
    ax = axes[0]
    ymin, ymax = 1.0, 0.0
    extrap = []
    for cs in css:
        pts = sorted(series[cs])
        n = np.array([p[0] for p in pts], float)
        y = np.array([p[1] for p in pts])
        ymin, ymax = min(ymin, y.min()), max(ymax, y.max())
        lab = f'$c_s$ = {cs:g}' + ('' if len(pts) >= 3 else '  (< 3 points)')
        ax.plot(1.0/n**2, y, 'o-', color=colors[cs], label=lab, zorder=3,
                markeredgecolor=style.SURFACE, markeredgewidth=1.2)
        if len(pts) >= 3:
            di, p, _ = _fit(n, y)
            ymin = min(ymin, di)
            xs = np.linspace(0.0, 1.0/n[0]**2, 100)
            ax.plot(xs, di + (y[0] - di)*(xs*n[0]**2)**(p/2.0),
                    color=colors[cs], lw=1.1, ls=':', zorder=2)
            ax.plot([0.0], [di], marker='*', ms=15, color=colors[cs],
                    markeredgecolor=style.SURFACE, markeredgewidth=1.0,
                    linestyle='none', zorder=4)
            extrap.append((cs, di, p, colors[cs]))
    # 外挿値はまとめて 1 箇所に書く (星の位置に並べるとぶつかる)。
    # **2 つの cs が同じ値に来ることがこの図の主張**なので、並べて見せる。
    if extrap:
        lines = [f'$N_r\\to\\infty$:'] + [
            f'  {d:.4f}   ($c_s$={c:g},  $p$={pp:.1f})' for c, d, pp, _ in extrap]
        ax.annotate('\n'.join(lines), xy=(0.30, 0.02),
                    xycoords='axes fraction', fontsize=9, color=style.INK,
                    va='bottom', ha='left',
                    bbox=dict(boxstyle='round,pad=0.4', fc=style.SURFACE,
                              ec=style.GRID, lw=1.0))
    ax.axhline(PAPER_DR, color=style.REFERENCE, lw=1.4, ls='--', zorder=1)
    ax.annotate('Rempel (2006)  Table 1 col. 2 = 0.27',
                xy=(0.98, PAPER_DR), xycoords=('axes fraction', 'data'),
                xytext=(0, 6), textcoords='offset points',
                color=style.REFERENCE, fontsize=9, ha='right')
    ax.set_xlabel('$1/N_r^2$        (coarser grid to the right)')
    ax.set_ylabel(r'$(\Omega_{\rm eq}-\Omega_{\rm pole})/\Omega_0$')
    ax.set_title(r'(a) Convergence with resolution   ($\star$ : $N_r\to\infty$)')
    ax.set_xlim(-0.5e-5, 9.8e-5)
    ax.set_ylim(ymin - 0.012, ymax + 0.012)
    ax.legend(loc='upper left')
    # 上軸に N_r を出す
    top = ax.secondary_xaxis('top')
    ns = [108, 144, 216, 288]
    top.xaxis.set_major_locator(FixedLocator([1.0/n**2 for n in ns]))
    top.set_xticklabels([str(n) for n in ns], fontsize=8)
    top.set_xlabel('$N_r$', fontsize=9, labelpad=2)
    top.tick_params(colors=style.INK3)
    top.spines['top'].set_color(style.INK3)

    # --- (b) 固定 N での cs 依存の幅 --------------------------------------
    ax = axes[1]
    widths = []
    for nx in sorted({k[0] for k in tab}):
        got = {}
        for (kx, ky, kc), v in tab.items():
            if kx == nx and v['sat']:
                got[round(kc, 4)] = v['DR']
        if 0.10 in got and 0.30 in got:
            widths.append((nx, got[0.10] - got[0.30]))
    n = np.array([w[0] for w in widths], float)
    w = np.array([w[1] for w in widths])
    ref = w[0]*(n[0]/n)**2
    ax.plot(n, ref, color=style.INK3, lw=1.3, ls='--', zorder=2,
            label='$N_r^{-2}$ (2nd order)')
    ax.plot(n, w, 'o-', color=style.BLUE[450], zorder=3,
            markeredgecolor=style.SURFACE, markeredgewidth=1.2,
            label='measured')
    for nn, ww in zip(n, w):
        ax.annotate(f'{ww:+.3f}', xy=(nn, ww), xytext=(0, 10),
                    textcoords='offset points', ha='center',
                    fontsize=9, color=style.INK2, fontweight='bold')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.xaxis.set_major_locator(FixedLocator(list(n)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}'))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([0.01, 0.02, 0.05, 0.1]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_ylim(0.008, 0.16)
    ax.set_xlabel('$N_r$')
    ax.set_ylabel(r'DR($c_s$=0.10) $-$ DR($c_s$=0.30)')
    ax.set_title('(b) Dependence on artificial diffusion vanishes')
    ax.legend(loc='upper right')

    fig.suptitle('Double limit of the differential rotation '
                 '(resolution $\\times$ artificial diffusion)',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = os.path.join(outdir, 'convergence_dr.png')
    fig.savefig(out)
    print('wrote', out)


if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'out')
    os.makedirs(d, exist_ok=True)
    main(d)
