"""解像度を上げると何が動いて何が動かないか (運動学的参照解、60 年).

Rempel (2006) の格子 108x72 と 144x96 を**同じ 60 年**で並べる。

分かること:

* **差動回転だけが解像度で動く** (0.321 -> 0.296、二重極限は 0.266)
* **周期はほとんど動かない** (17.86 -> 17.93 年、0.4 パーセント)。
  代わりに同じランの中で 16.8 -> 18.1 年と時間とともに伸びる
* max|B_phi| も E_B も 60 年で頭打ちになっていない。**運動学的ランは
  ローレンツ力による飽和がないので、alpha クエンチングだけでは効きが遅い**

トロイダル磁場は符号のある量なので diverging で塗る。解像度は順序量なので
青の単一色ランプにする (style.py の方針)。
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import style                                            # noqa: E402

YR = 3.156e7
PAPER = dict(period=19.0, Bph=1.28, DR=0.27)
DR_CONVERGED = 0.266          # 二重極限 (convergence.py --extrap)

# (ラベル, つなぐタグの並び)
RUNS = [('108x72', ['p_kin']),
        ('144x96', ['res144_kin', 'res144_kin_b'])]


def load(tags):
    """分割されたランを時刻順につなぐ (重複時刻は落とす)."""
    hs, bs, th = [], [], None
    for tag in tags:
        d = np.load(f'results_rempel/{tag}/dynamo.npz')
        hs.append(d['hist'])
        bs.append(d['butter'])
        th = d['th']
    h = np.vstack(hs)
    b = np.vstack(bs)
    o = np.argsort(h[:, 0])
    h, b = h[o], b[o]
    keep = np.concatenate(([True], np.diff(h[:, 0]) > 0))
    return h[keep], b[keep], th


def cycles(t, b):
    """符号反転から (サイクル中心時刻, 周期) を返す."""
    idx = np.where(np.diff(np.sign(b)) != 0)[0]
    tz = t[idx]
    if len(tz) < 2:
        return np.array([]), np.array([])
    return 0.5*(tz[:-1] + tz[1:]), 2*np.diff(tz)


def main(outdir):
    style.apply()
    os.chdir(ROOT)
    colors = style.ordinal_blues(len(RUNS))

    fig, axes = plt.subplots(4, 1, figsize=(8.8, 9.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.5, 1, 1, 1],
                                              hspace=0.14))

    # --- (a) 蝶形図 (細かいほうだけ) ------------------------------------
    h, b, th = load(RUNS[-1][1])
    t = h[:, 0]/YR
    lat = 90.0 - np.degrees(th)
    B = b.T*1e-4                                   # G -> T
    vmax = np.abs(B).max()
    ax = axes[0]
    im = ax.pcolormesh(t, lat, B, cmap=style.DIVERGING, shading='auto',
                       vmin=-vmax, vmax=vmax, rasterized=True)
    cb = fig.colorbar(im, ax=ax, pad=0.012, aspect=16)
    cb.set_label(r'$B_\Phi$ at $r=0.735\,R_\odot$  [T]', fontsize=9)
    ax.set_ylabel('latitude  [deg]')
    ax.set_title(f'(a)  butterfly diagram, {RUNS[-1][0]} kinematic',
                 loc='left')
    ax.set_ylim(0, 90)
    ax.grid(False)

    # --- (b) max|B_phi| --------------------------------------------------
    ax = axes[1]
    for (label, tags), c in zip(RUNS, colors):
        h, _, _ = load(tags)
        ax.plot(h[:, 0]/YR, np.abs(h[:, 1]), color=c, lw=1.4, label=label)
    ax.axhline(PAPER['Bph'], color=style.REFERENCE, lw=1.2, ls='--')
    ax.annotate('Rempel (2006): 1.28 T', xy=(1.0, PAPER['Bph']),
                xytext=(1.5, PAPER['Bph'] + 0.06), color=style.REFERENCE,
                fontsize=8.5)
    ax.set_ylabel(r'max$|B_\Phi|$  [T]')
    ax.set_title('(b)  toroidal field — still growing at 60 yr', loc='left')
    ax.legend(loc='lower right', ncol=2)

    # --- (c) 差動回転 -----------------------------------------------------
    ax = axes[2]
    for (label, tags), c in zip(RUNS, colors):
        h, _, _ = load(tags)
        ax.plot(h[:, 0]/YR, h[:, 4], color=c, lw=1.8, label=label)
    for y, txt, ls in ((PAPER['DR'], 'Rempel (2006): 0.27', '--'),
                       (DR_CONVERGED, r'our $N\to\infty$: 0.266', ':')):
        ax.axhline(y, color=style.REFERENCE, lw=1.2, ls=ls)
    ax.annotate('Rempel (2006): 0.27   /   our $N\\to\\infty$: 0.266',
                xy=(1.0, PAPER['DR']), xytext=(1.5, 0.235),
                color=style.REFERENCE, fontsize=8.5)
    ax.set_ylabel(r'$(\Omega_{\rm eq}-\Omega_{\rm pole})/\Omega_0$')
    ax.set_title('(c)  differential rotation — the one quantity resolution '
                 'moves', loc='left')
    ax.set_ylim(0.22, 0.35)

    # --- (d) サイクルごとの周期 -------------------------------------------
    ax = axes[3]
    for (label, tags), c in zip(RUNS, colors):
        h, b, th = load(tags)
        t = h[:, 0]/YR
        lat40 = np.argmin(abs(th - np.radians(50)))
        tc, per = cycles(t, b[:, lat40])
        ax.plot(tc, per, 'o-', color=c, lw=1.8, ms=5, label=label)
    ax.axhline(PAPER['period'], color=style.REFERENCE, lw=1.2, ls='--')
    ax.annotate('Rempel (2006): 19 yr', xy=(1.0, PAPER['period']),
                xytext=(1.5, 18.72), color=style.REFERENCE, fontsize=8.5)
    ax.set_ylabel('cycle period  [yr]')
    ax.set_xlabel('time  [yr]')
    ax.set_title('(d)  period — set by saturation, not by resolution',
                 loc='left')
    ax.set_ylim(16.3, 19.4)
    ax.set_xlim(0, 60)

    for a in axes[1:]:
        style.despine(a)

    out = os.path.join(outdir, 'resolution_60yr.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'  {out}')
    return out


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(ROOT, 'doc', 'source', '_static', 'figures'))
