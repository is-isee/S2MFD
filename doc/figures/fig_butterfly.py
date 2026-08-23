"""蝶形図とサイクルの時系列 (Rempel 2006 図 3・図 4 に対応).

トロイダル磁場は**符号のある量**なので diverging (青 ↔ 赤、中点は灰色) で
塗る。虹色は使わない。
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, 'run_paris')]
import style                                            # noqa: E402
import table1                                           # noqa: E402

YR = 3.156e7


def _cycles(t, b, n_cycles=3.0, period_guess=18.0):
    """末尾の n_cycles 分の区間を返す."""
    t0 = t[-1] - n_cycles*period_guess
    return t >= max(t0, t[0])


def main(outdir, tag='v2_a125', alpha0=12.5):
    style.apply()
    os.chdir(ROOT)
    d = np.load(f'results_rempel/{tag}/dynamo.npz')
    h, th, bu = d['hist'], d['th'], d['butter']
    t = h[:, 0]/YR
    lat = 90.0 - np.degrees(th)          # 余緯度 -> 緯度
    sel = _cycles(t, bu)
    tt = t[sel] - t[sel][0]
    B = bu[sel].T*1e-4                   # G -> T

    per, ncyc = table1.cycle_period(t, bu[:, np.argmin(abs(th - np.radians(50)))])

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.5, 1, 1],
                                              hspace=0.16))

    # --- (a) 蝶形図 ------------------------------------------------------
    ax = axes[0]
    vmax = np.abs(B).max()
    im = ax.pcolormesh(tt, lat, B, cmap=style.DIVERGING, shading='auto',
                       vmin=-vmax, vmax=vmax, rasterized=True)
    cb = fig.colorbar(im, ax=ax, pad=0.015, aspect=18)
    cb.set_label(r'$B_\Phi$  at  $r=0.735\,R_\odot$   [T]', fontsize=9)
    cb.outline.set_visible(False)
    # 論文: 緯度 50 度から始まり 40 度でピーク
    for y0, lab in ((50.0, 'onset  50$^\\circ$'), (40.0, 'peak  40$^\\circ$')):
        ax.axhline(y0, color=style.INK, lw=1.0, ls=':', alpha=0.55)
        ax.annotate(lab, xy=(0.995, y0), xycoords=('axes fraction', 'data'),
                    xytext=(0, 3), textcoords='offset points',
                    ha='right', fontsize=8, color=style.INK2)
    ax.set_ylabel('latitude [deg]')
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 30, 60, 90])
    ax.set_title(f'(a) Butterfly diagram   ({tag},  '
                 f'$\\alpha_0$ = {alpha0/100:g} m s$^{{-1}}$,  '
                 f'period = {per:.1f} yr)')

    # --- (b) 表面の径方向磁場と 0.735R の磁場 -----------------------------
    ax = axes[1]
    ax.plot(tt, h[sel, 3]*1e-4, color=style.BLUE[450],
            label=r'$B_\Phi$  (0.735 $R_\odot$, equator)')
    ax.axhline(0, color=style.INK3, lw=0.8)
    ax.set_ylabel(r'$B_\Phi$  [T]')
    ax.legend(loc='upper right')
    ax.set_title('(b) Toroidal field at the base of the convection zone')

    # --- (c) トーショナル振動 --------------------------------------------
    ax = axes[2]
    for col, lab, c in ((5, r'pole  (90$^\circ$)', style.BLUE[450]),
                        (6, r'60$^\circ$ latitude', style.ORANGE_RAMP[2])):
        y = table1.detrend(t[sel], h[sel, col], 18.0)
        n = (len(tt) - len(y))//2
        ax.plot(tt[n:n+len(y)], y, color=c, label=lab)
    ax.axhline(0, color=style.INK3, lw=0.8)
    ax.set_ylabel(r'$\Omega-\bar\Omega$  [nHz]')
    ax.set_xlabel('time [yr]')
    ax.legend(loc='upper right', ncol=2)
    ax.set_title(r'(c) Torsional oscillation at $r=0.985\,R_\odot$')

    fig.suptitle(f'Nonkinematic flux-transport dynamo  —  {tag}',
                 fontsize=12, fontweight='bold', y=0.995)
    out = os.path.join(outdir, f'butterfly_{tag}.png')
    fig.savefig(out)
    print('wrote', out)


if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'out')
    os.makedirs(d, exist_ok=True)
    main(d)
