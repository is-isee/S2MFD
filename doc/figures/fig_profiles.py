"""背景プロファイルと緩和した参照モデル (Rempel 2006 図 1・図 2 に対応).

差動回転は符号のある摂動なので diverging、拡散係数などの正の量は
単一色の sequential。緯度は順序量なので青のランプ。
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import style                                            # noqa: E402
import S2MFD                                            # noqa: E402
from S2MFD.stratification import Stratification         # noqa: E402


def main(outdir, state='results_rempel/s288x192_cs030/state.npz',
         nx=288, ny=192):
    style.apply()
    os.chdir(ROOT)
    cfg = S2MFD.build_cfg('parameters/rempel06_paper.py', ix=nx, jx=ny,
                          sld_cs_factor=0.30)
    grid = S2MFD.Grid.from_cfg(cfg)
    strat = Stratification(cfg, grid)
    setup = S2MFD.Setup(cfg, grid)
    m = grid.margin
    sl = (slice(m, grid.ixg-m), slice(m, grid.jxg-m))
    rr = grid.rr[m:grid.ixg-m]/cfg.RSUN
    th = grid.th[m:grid.jxg-m]
    d = np.load(state)
    om1 = d['om1'][sl]
    vth = d['vth'][sl]
    om_nhz = (cfg.om0 + om1)/(2*np.pi)*1e9

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))

    # --- (a) Omega の等値線 (子午面) --------------------------------------
    ax = axes[0, 0]
    X = np.outer(rr, np.sin(th)); Y = np.outer(rr, np.cos(th))
    lv = np.linspace(np.floor(om_nhz.min()/10)*10,
                     np.ceil(om_nhz.max()/10)*10, 25)
    cf = ax.contourf(X, Y, om_nhz, levels=lv, cmap=style.SEQUENTIAL)
    ax.contour(X, Y, om_nhz, levels=lv[::3], colors=style.INK, linewidths=0.6,
               alpha=0.5)
    cb = fig.colorbar(cf, ax=ax, pad=0.02, aspect=22)
    cb.set_label(r'$\Omega/2\pi$  [nHz]', fontsize=9)
    cb.outline.set_visible(False)
    ax.plot(0.71*np.sin(th), 0.71*np.cos(th), color=style.INK, lw=1.2,
            ls='--')
    ax.annotate(r'$r_{\rm bc}=0.71\,R_\odot$', xy=(0.62, 0.88),
                fontsize=8, color=style.INK2,
                bbox=dict(boxstyle='round,pad=0.25', fc=style.SURFACE,
                          ec='none', alpha=0.85))
    ax.set_aspect('equal'); ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
    ax.set_xlabel('$r/R_\\odot$'); ax.set_ylabel('$r/R_\\odot$')
    ax.set_title('(a) Differential rotation')
    ax.grid(False)

    # --- (b) Omega(r) を緯度ごとに ---------------------------------------
    ax = axes[0, 1]
    lats = [0, 15, 30, 45, 60, 90]
    cols = style.ordinal_blues(len(lats))
    for la, c in zip(lats, cols):
        j = np.argmin(abs(th - np.radians(90 - la)))
        ax.plot(rr, om_nhz[:, j], color=c, label=f'{la}$^\\circ$')
    ax.axvline(0.71, color=style.REFERENCE, lw=1.2, ls='--')
    ax.annotate(r'$r_{\rm bc}$', xy=(0.71, 0.02), xycoords=('data',
                'axes fraction'), xytext=(4, 0), textcoords='offset points',
                fontsize=9, color=style.REFERENCE)
    ax.set_xlabel('$r/R_\\odot$')
    ax.set_ylabel(r'$\Omega/2\pi$  [nHz]')
    ax.set_title('(b) Radial profiles by latitude')
    ax.legend(title='latitude', ncol=2, loc='lower right')

    # --- (c) 子午面流 ------------------------------------------------------
    ax = axes[1, 0]
    v = vth/100.0                        # cm/s -> m/s
    vmax = np.abs(v).max()
    cf = ax.contourf(X, Y, v, levels=np.linspace(-vmax, vmax, 25),
                     cmap=style.DIVERGING)
    cb = fig.colorbar(cf, ax=ax, pad=0.02, aspect=22)
    cb.set_label(r'$v_\theta$  [m s$^{-1}$]'
                 '\n(red: equatorward,  blue: poleward)', fontsize=9)
    cb.outline.set_visible(False)
    ax.plot(0.71*np.sin(th), 0.71*np.cos(th), color=style.INK, lw=1.2,
            ls='--')
    ax.set_aspect('equal'); ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
    ax.set_xlabel('$r/R_\\odot$'); ax.set_ylabel('$r/R_\\odot$')
    ax.set_title('(c) Meridional flow')
    ax.grid(False)

    # --- (d) 拡散係数のプロファイル ----------------------------------------
    ax = axes[1, 1]
    et = np.asarray(setup.et)
    et = et[m:grid.ixg-m] if et.ndim == 1 else et[m:grid.ixg-m, 0]
    ax.semilogy(rr, et*1e-4, color=style.BLUE[450], label=r'$\eta_t$')
    ax.semilogy(rr, np.asarray(setup.nu_dif)[m:grid.ixg-m]*1e-4,
                color=style.ORANGE_RAMP[2], label=r'$\nu_t$ (diffusive)')
    ax.semilogy(rr, np.asarray(setup.nu_lam)[m:grid.ixg-m]*1e-4,
                color=style.ORANGE_RAMP[2], ls='--',
                label=r'$\nu_t$ ($\Lambda$ effect)')
    ax.axvline(0.71, color=style.REFERENCE, lw=1.2, ls='--')
    ax.set_xlabel('$r/R_\\odot$')
    ax.set_ylabel(r'diffusivity  [m$^2$ s$^{-1}$]')
    ax.set_title('(d) Turbulent diffusivities  (Rempel 2006 eqs. 13-15, 27)')
    ax.legend(loc='lower right')

    fig.suptitle('Reference model (no magnetic field), '
                 f'{nx}$\\times${ny},  $c_s$ = 0.30',
                 fontsize=12, fontweight='bold', y=0.995)
    fig.tight_layout()
    out = os.path.join(outdir, 'reference_model.png')
    fig.savefig(out)
    print('wrote', out)


if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'out')
    os.makedirs(d, exist_ok=True)
    main(d)
