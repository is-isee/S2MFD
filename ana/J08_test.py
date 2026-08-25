"""Jouve et al. (2008) ベンチマーク比較用の時系列プロット。

使い方:
    python J08_test.py [datadir]   # 既定は ../data/
"""
import matplotlib.pyplot as plt
import numpy as np

from ana_common import (get_datadir, load_run, radial_index,
                        colat_index, surface_index)

datadir = get_datadir()
run = load_run(datadir)
cfg, grid, timet, tau_diff = run.cfg, run.grid, run.timet, run.tau_diff
Bpht, Brrt = run.Bpht, run.Brrt

# 添字はゴーストを踏まないヘルパで取る (2026-08-23)。ana_common 参照。
i07 = radial_index(grid, 0.7*cfg.RSUN)
isurf = surface_index(grid)
Bpht0 = Bpht[i07, colat_index(grid, 30.0), :]
Brrt0 = Brrt[isurf, colat_index(grid, 60.0), :]

Bpht0_sign = np.sign(Bpht0)
Bpht0_sign_diff = np.diff(Bpht0_sign)

Brrt0_sign = np.sign(Brrt0)
Brrt0_sign_diff = np.diff(Brrt0_sign)
ne_r = np.where(Brrt0_sign_diff == +2)[0][-1]


ns = np.where(Bpht0_sign_diff == +2)[0][-2]
ne = np.where(Bpht0_sign_diff == +2)[0][-1]

plt.clf()
plt.close('all')
fig = plt.figure('J08_test',figsize=(6,10))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

timeu = (timet[ns:ne]-timet[ns])/tau_diff
timeur = (timet[ns:ne_r]-timet[ns])/tau_diff
Bpht0u = Bpht0[ns:ne]
Brrt0u = Brrt0[ns:ne]
nw = ne - ns
nm = np.argmax(Bpht0u)

ax1.plot(timeu,Bpht0u)
ax2.plot(timeu,Brrt0u)

ax1.set_ylabel(r'$B_\phi$: $r=0.7R_\odot$, $\theta=30^\circ$')
ax2.set_ylabel(r'$B_r$: $r=R_\odot$, $\theta=60^\circ$')

ax2.set_xlabel(r't/$\tau_\mathrm{diff}$')

fig.tight_layout()

print('Cycle time = ',timeu[-1])
print('Cycle time = ',timeur[-1])
print('Period(year) = ',timeu[-1]*tau_diff/86400/365,'year')
print('Max(Bph) =',np.max(Bpht0u))
plt.show()
