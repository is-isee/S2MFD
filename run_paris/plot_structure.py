"""磁場の空間構造を Rempel 2006 図 3-5 と比べるための図。

  (a) 蝶形図: r=0.735 RSUN の B_phi (緯度 x 時間)
  (b) r-theta 断面の B_phi (最終時刻)
  (c) r=0.735 RSUN での緯度プロファイル (サイクル最大時)
  (d) 磁場が最大の緯度での動径プロファイル
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'tests')
import matplotlib; matplotlib.use('Agg')
matplotlib.rcParams['axes.unicode_minus']=False
import matplotlib.pyplot as plt
np.seterr(all='ignore')
import S2MFD
from S2MFD.stratification import Stratification
from S2MFD.physics import poloidal_mag
from conftest import make_cfg, make_grid
YR=3.156e7
tag = sys.argv[1]; alpha0 = float(sys.argv[2])
d = np.load(f'results_rempel/{tag}/dynamo.npz')
h, bu, th, rr = d['hist'], d['butter'], d['th'], d['rr']
t = h[:,0]/YR
fs = np.load(f'results_rempel/{tag}/final_state.npz')
cfg = make_cfg('parameters/rempel06_paper.py', alpha0=alpha0, magnetic_buoyancy=1,
               sld_cs_factor=0.30, alpha_quenching=False)
grid = make_grid(cfg); Stratification(cfg,grid); S2MFD.Setup(cfg,grid)
m = grid.margin
Bph2d = fs['Bph'][m:-m, m:-m]*1e-4          # T
lat = 90.0 - np.degrees(th)
RS = rr/cfg.RSUN

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
# (a) 蝶形図 (末尾 40 年)
sel = t >= t[-1]-40
v = np.abs(bu[sel]).max()*1e-4
im = ax[0,0].pcolormesh(t[sel], lat, (bu[sel]*1e-4).T, cmap='RdBu_r',
                        vmin=-v, vmax=v, shading='auto')
ax[0,0].set_xlabel('time [yr]'); ax[0,0].set_ylabel('latitude [deg]')
ax[0,0].set_title(f'(a) butterfly: $B_\\varphi$ at $r=0.735R_\\odot$  [T]')
ax[0,0].axhline(50, color='k', ls=':', lw=0.8); ax[0,0].axhline(40, color='k', ls='--', lw=0.8)
ax[0,0].set_ylim(0, 90); plt.colorbar(im, ax=ax[0,0])
# (b) r-theta 断面
v2 = np.abs(Bph2d).max()
im2 = ax[0,1].pcolormesh(RS, lat, Bph2d.T, cmap='RdBu_r', vmin=-v2, vmax=v2, shading='auto')
ax[0,1].set_xlabel('$r/R_\\odot$'); ax[0,1].set_ylabel('latitude [deg]')
ax[0,1].set_title(f'(b) $B_\\varphi$ at $t={t[-1]:.0f}$ yr  [T]')
ax[0,1].axvline(0.735, color='k', ls='--', lw=0.8); ax[0,1].set_ylim(0, 90)
plt.colorbar(im2, ax=ax[0,1])
# (c) 緯度プロファイル (|B| が最大の時刻)
k = np.argmax(np.abs(bu).max(axis=1))
ax[1,0].plot(lat, bu[k]*1e-4, lw=2, label=f'this run (t={t[k]:.1f} yr)')
ax[1,0].set_xlabel('latitude [deg]'); ax[1,0].set_ylabel('$B_\\varphi$ [T]')
ax[1,0].set_title('(c) latitude profile at $0.735R_\\odot$ (cycle max)')
ax[1,0].axvline(40, color='r', ls='--', lw=0.8, label='paper: peak 40 deg')
ax[1,0].axvline(50, color='r', ls=':', lw=0.8, label='paper: onset 50 deg')
ax[1,0].set_xlim(0, 90); ax[1,0].legend(fontsize=8); ax[1,0].grid(alpha=0.3)
# (d) 動径プロファイル
jmax = np.argmax(np.abs(Bph2d).max(axis=0))
ax[1,1].plot(RS, Bph2d[:, jmax], lw=2, label=f'lat={lat[jmax]:.0f} deg')
ax[1,1].set_xlabel('$r/R_\\odot$'); ax[1,1].set_ylabel('$B_\\varphi$ [T]')
ax[1,1].set_title('(d) radial profile at peak latitude')
ax[1,1].axvline(0.735, color='r', ls='--', lw=0.8, label='paper: peak 0.735')
ax[1,1].axvline(0.71, color='gray', ls=':', lw=0.8, label='base of CZ 0.71')
ax[1,1].legend(fontsize=8); ax[1,1].grid(alpha=0.3)
fig.suptitle(f'{tag}  ($\\alpha_0={alpha0/100:.3f}$ m/s)   '
             f'max$|B_\\varphi|$={np.abs(h[:,1]).max():.3f} T', fontsize=12)
fig.tight_layout()
out=f'results_rempel/{tag}_structure.png'; fig.savefig(out, dpi=110)
print(f"wrote {out}")
print(f"  蝶形図のピーク緯度: {lat[np.argmax(np.abs(bu[k]))]:.0f} deg (論文 40 deg)")
print(f"  動径のピーク: r={RS[np.argmax(np.abs(Bph2d[:,jmax]))]:.3f} R (論文 0.735)")
