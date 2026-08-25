"""蝶形図 (butterfly diagram) を描くスクリプト。

使い方:
    python butterfly_diagram.py [datadir]   # 既定は ../data/
"""
import matplotlib.pyplot as plt
import numpy as np

from ana_common import (get_datadir, load_run, radial_index,
                        colat_index, surface_index,
                        physical_colat_deg, physical_slice)

datadir = get_datadir()
run = load_run(datadir)
cfg, grid, timet, tau_diff = run.cfg, run.grid, run.timet, run.tau_diff
Bpht, Brrt = run.Bpht, run.Brrt

# 添字はゴーストを踏まないヘルパで取る (2026-08-23)。
# 以前は表面を Brrt[-2] としていたが、これは margin=1 でしか最外物理セルに
# ならない (margin=2 の Rempel 設定ではゴーストセル)。半径も
# 1+argmin(...) と 1 セル外側を指していた。
i07 = radial_index(grid, 0.7*cfg.RSUN)
isurf = surface_index(grid)
jth = physical_slice(grid)[1]
th_deg = physical_colat_deg(grid)

Bpht_c = Bpht[i07, jth, :]
Brrt_s = Brrt[isurf, jth, :]

Bpht0 = Bpht[i07, colat_index(grid, 30.0), :]
Brrt0 = Brrt[isurf, colat_index(grid, 60.0), :]
print(f"B_phi: r = {grid.rr[i07]/cfg.RSUN:.4f} R, "
      f"B_r: r = {grid.rr[isurf]/cfg.RSUN:.4f} R (最外物理セル)")

Bpht0_sign = np.sign(Bpht0)
Bpht0_sign_diff = np.diff(Bpht0_sign)

ns = np.where(Bpht0_sign_diff == +2)[0][-3]
ne = np.where(Bpht0_sign_diff == +2)[0][-1]

timeu = (timet[ns:ne]-timet[ns])/tau_diff
time_year = (timet[ns:ne]-timet[ns])/86400/365
plt.clf()
plt.close('all')
fig = plt.figure('Butterfly Diagram',figsize=(10,10))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

# butterfly diagram
# B_0: 磁場の規格化 (クエンチング B/(1+B^2) の単位磁場) [G]
B_0 = 4.e4
c1 = ax1.pcolormesh(time_year, th_deg, B_0*Bpht_c[:,ns:ne], cmap='bwr', shading='auto')
c2 = ax2.pcolormesh(time_year, th_deg, B_0*Brrt_s[:,ns:ne], cmap='bwr', shading='auto')

# カラーバーを追加
fig.colorbar(c1, ax=ax1, orientation='vertical').set_label(r'$B_\phi$ (G)')
fig.colorbar(c2, ax=ax2, orientation='vertical').set_label(r'$B_r$ (G)')

# sunspot
Bpht_lim = Bpht_c[:,ns:ne]
mask_p = Bpht_lim > 1.0
mask_m = Bpht_lim < -1.0
ax1.contourf(timeu, th_deg, mask_p, levels=[0.5, 1.5], colors=['black'])
ax1.contourf(timeu, th_deg, mask_m, levels=[0.5, 1.5], colors=['black'])

ax1.set_ylabel(r'$B_\phi$: $r=0.7R_\odot$')
ax2.set_ylabel(r'$B_r$: 最外物理セル')

ax2.set_xlabel('t(year)')

fig.tight_layout()
plt.show()
